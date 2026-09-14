---
title: "Search and Sorting Algorithms: Binary Search, Hash Tables, and Sorting Implementations"
title_zh: 查找与排序算法：二分查找、散列表与排序实现
date: 2024-12-18 21:37:43 +0800
categories:
  - Programming
  - CS Foundations
tags:
  - Algorithms
  - Search
  - Sorting
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers sequential search, binary search, binary search trees, hash tables, sorting algorithms, and related Python implementations.
description: Covers sequential search, binary search, binary search trees, hash tables, sorting algorithms, and related Python implementations.
excerpt_zh: 整理顺序查找、二分查找、二叉搜索树、散列表、排序算法及其 Python 实现。
permalink: /blog/2024/12/18/search-and-sorting-algorithms/
lang: en
translation_key: 2024-12-18-search-and-sorting-algorithms
translation_status: machine
translation_source_hash: 233d662f410963dbf442c24f12cd43d06a86d16f0c2cfe962930e93775d3436b
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

## Search

This article can be read together with [Introduction to Data Structures: Linear Lists, Trees, Graphs, and Search](/en/blog/2025/05/12/data-structures-introduction/) and [Algorithm Design and Analysis: Divide and Conquer, Dynamic Programming, and Graph Algorithms](/en/blog/2025/05/13/algorithm-design-and-analysis/). The topics overlap, but each article has a different focus.

Searching is a common operation in programs and an unavoidable topic when learning data structures. The question is straightforward: given a set of data and a key, how can we find the corresponding record quickly? The data size, whether the data is ordered, and whether insertion or deletion is required all affect the choice of search method and data structure.

### Search basics

A collection of records to be searched is called a **search table**.

A **key** is a field of a data element that can identify the element or a field in a record.

If a key uniquely identifies a record, it is a **primary key**. Otherwise, it is a **secondary key**.

Searching means finding the record associated with a given key. A successful search usually returns the record or its position; an unsuccessful search returns `None` or a designated invalid index.

Based on whether the data can be modified during the operation, search tables are divided into static and dynamic search tables:

- **Static search table**: supports queries only; records are not inserted or deleted.
- **Dynamic search table**: supports insertion or deletion while it is being searched.

Choosing an appropriate data structure can improve search performance. A linear table is often enough for static search. A binary search tree is useful when the table changes dynamically. A hash table is a good choice when direct key-based lookup is needed and range queries are not.

### Searching a linear table

#### Sequential search

When elements are stored in an unordered linear table, there is no ordering information to exploit. The direct approach is to compare elements from the beginning to the end. The code below defines a simple sequential table and implements both ordinary search and sentinel search.

```python
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class SequenceTable(Generic[T]):
    """A sequential table backed by a Python list."""

    items: list[T]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> T:
        return self.items[index]


def linear_search(table: SequenceTable[int], target: int) -> int:
    """Return target's index, or -1 when it is not present."""
    for index, value in enumerate(table.items):
        if value == target:
            return index
    return -1


def linear_search_with_sentinel(table: SequenceTable[int], target: int) -> int:
    """Temporarily append a sentinel to avoid one boundary check."""
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

Sequential search does not require ordered data and has $O(n)$ time complexity. A sentinel reduces a boundary check, but it does not change the worst-case complexity.

#### Searching an ordered table

If elements are ordered by a key, the ordering can be used to reduce the search range. Binary search, interpolation search, and Fibonacci search all rely on an ordered table.

#### Binary search

Binary search checks the middle element of the current interval. If it is greater than the target, the search continues in the left half; if it is smaller, the search continues in the right half. The implementation below uses the closed interval `[left, right]`.

```python
from dataclasses import dataclass


@dataclass
class OrderedTable:
    """An ordered table containing integers in ascending order."""

    items: list[int]

    def __post_init__(self) -> None:
        if self.items != sorted(self.items):
            raise ValueError("items must be sorted in ascending order")

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

Binary search has $O(\log n)$ time complexity, provided that the table is ordered and supports efficient indexed access.

#### Interpolation search

Binary search always chooses the middle position. For an ordered table whose keys are distributed fairly uniformly, interpolation search estimates the position from the target's relative position between the first and last keys:

$$
mid = low + \frac{(high-low)(key-a[low])}{a[high]-a[low]}
$$

When the first and last elements are equal, this formula cannot be used because it would divide by zero. Interpolation search is suitable for uniformly distributed keys; with a skewed distribution, it is not necessarily better than binary search.

```python
from dataclasses import dataclass


@dataclass
class InterpolationTable:
    """An ascending integer table with indexed access for interpolation search."""

    items: list[int]

    def __post_init__(self) -> None:
        if self.items != sorted(self.items):
            raise ValueError("items must be sorted in ascending order")


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

#### Fibonacci search

Fibonacci search uses the Fibonacci sequence to choose a split position. The sequence satisfies $F(n)=F(n-1)+F(n-2)$, so the search interval shrinks according to the same proportions. The implementation below uses the last element to fill a logical array when necessary, while returning an index from the original table.

```python
from dataclasses import dataclass


@dataclass
class FibonacciTable:
    """An ascending integer table that is not modified during the search."""

    items: list[int]

    def __post_init__(self) -> None:
        if self.items != sorted(self.items):
            raise ValueError("items must be sorted in ascending order")


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

### Linear indexing

When the data is large and cannot conveniently be kept as one globally ordered sequence, an index can be built. The index associates keys with record positions. A search first locates the index entry and then finds the record in the original data.

Index structures are commonly divided into **linear indexes, tree indexes, and multilevel indexes**. This article focuses on linear indexes.

### Binary search tree

A dynamic search table must support search as well as convenient insertion and deletion. A binary search tree (BST) builds this structure by comparing keys: smaller values go into the left subtree and larger values go into the right subtree. An in-order traversal of a BST produces an ordered sequence.

A binary search tree has these properties:

- If the left subtree is not empty, every value in it is smaller than the root value.
- If the right subtree is not empty, every value in it is larger than the root value.
- The left and right subtrees are also binary search trees.

The purpose of a BST is not merely to sort data, but to support search, insertion, and deletion through the tree structure. The code below defines the node and tree first, then implements search, insertion, minimum lookup, and deletion.

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BSTNode:
    """A BST node containing one key and references to two subtrees."""

    key: int
    left: BSTNode | None = None
    right: BSTNode | None = None


class BinarySearchTree:
    """A binary search tree built from BSTNode objects."""

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

Let $h$ be the height of the BST. Search, insertion, and deletion each follow at most one root-to-leaf path, so their time complexity is $O(h)$. When the tree is reasonably balanced, $h=O(\log n)$; when the tree degenerates, $h=O(n)$.

| Operation | Balanced | Worst case |
| --- | --- | --- |
| Search | $O(\log n)$ | $O(n)$ |
| Insertion | $O(\log n)$ | $O(n)$ |
| Deletion | $O(\log n)$ | $O(n)$ |
| Minimum or maximum lookup | $O(\log n)$ | $O(n)$ |

An in-order traversal visits every node, so its time complexity is $O(n)$ regardless of balance. If keys are inserted in ascending or descending order, each node may have only one child and the BST degenerates into a linked list. Search, insertion, and deletion then become linear-time operations, and a recursive implementation may also exceed the call-stack limit. Balanced trees such as AVL trees use rotations to keep the height small and address this problem.

### AVL tree

An AVL tree requires the heights of the left and right subtrees of every node to differ by at most 1. The difference is called the balance factor:

$$
BF = height(left) - height(right)
$$

In a balanced state, BF can only be -1, 0, or 1. After insertion or deletion, if the absolute balance factor of a node exceeds 1, the smallest unbalanced subtree must be adjusted.

AVL trees use rotations for adjustment. The usual cases are LL, RR, LR, and RL: LL uses a right rotation, RR uses a left rotation, LR uses a left rotation followed by a right rotation, and RL uses a right rotation followed by a left rotation.

The code below defines `AVLNode` with a height field, then implements height updates, rotations, and rebalancing.

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AVLNode:
    """An AVL node with a key, two children, and its subtree height."""

    key: int
    left: AVLNode | None = None
    right: AVLNode | None = None
    height: int = 1


class AVLTree:
    """A binary search tree that maintains balance through rotations."""

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
            return cls.rotate_right(node)  # LL or LR
        if factor < -1:
            if cls.balance_factor(node.right) > 0:
                node.right = cls.rotate_right(node.right)  # RL
            return cls.rotate_left(node)  # RR or RL
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

When rebalancing is placed in the recursive unwind phase of insertion and deletion, every ancestor can update its height and check its balance factor. AVL tree search, insertion, and deletion remain $O(\log n)$.

### Multiway search trees: B-trees

A binary tree allows at most two children per node. When data is large and primarily stored on disk, tree height directly affects the number of disk accesses. Disk access is usually slower than memory and cache access. A multiway search tree stores several keys in a node and has several children, reducing the tree height.

The following sections introduce 2-3 trees, 2-3-4 trees, B-trees, and B+ trees.

#### 2-3 trees

A 2-3 tree has two kinds of nodes:

- A 2-node contains one key and two children, or no children.
- A 3-node contains two ordered keys and three children, or no children.

For a 2-node, the left subtree is smaller than the key and the right subtree is larger. For a 3-node, the left, middle, and right subtrees contain values smaller than the smaller key, between the two keys, and larger than the larger key, respectively.

All leaves of a 2-3 tree must be at the same level. Insertion occurs at a leaf: an empty tree receives a 2-node; inserting into a 2-node can produce a 3-node; inserting into a 3-node requires splitting it and promoting the middle key. Deletion from a 3-node is relatively simple, while deletion from a 2-node may require borrowing a key or merging nodes.

#### 2-3-4 trees

A 2-3-4 tree extends the 2-3 tree idea. A node can contain up to three keys and have four children. Its insertion and deletion rules are more involved, but all leaves are still kept at the same level.

#### B-trees

A B-tree is a balanced multiway search tree. 2-3 trees and 2-3-4 trees can be viewed as special cases. The order of a B-tree usually describes the maximum number of children a node may have. In practice, the order can be chosen based on page size, record size, and available memory. A larger order generally produces a shorter tree; keeping the root in memory can reduce the number of external-memory accesses.

#### B+ trees

A B+ tree is an improved form of a B-tree. Internal nodes mainly store indexes, while records are usually stored in leaf nodes. The leaves are also linked in order, which makes B+ trees particularly suitable for range queries and sequential scans.

### Hash table overview

Sequential and tree searches both rely on key comparisons. A hash table attempts to calculate a storage position directly from a key:

$$
address = h(key)
$$

The function $h$ is a **hash function**, and the continuous storage area is the **hash table**. Ideally, different keys would map to different addresses. In practice, multiple keys mapping to one address is unavoidable; this is a **collision**, and the different keys involved are called **synonyms**.

A hash table is both a storage structure and a search structure. It does not emphasize a logical ordering between elements; it is designed for key-based lookup. It is therefore not suitable for cases where one key maps to many records or for range queries.

### Constructing hash functions

A useful hash function should be simple to compute and should distribute addresses as evenly as possible to reduce collisions. Common approaches include the following.

#### Direct addressing

Use the key itself, or a linear function of the key, as the address. For example, age can be used directly as an address when counting people by age, and year can be used when counting births by year.

This method is simple and easy to reason about. It can avoid collisions when keys are unique, but it requires the key range to be known in advance and may waste space.

#### Digit analysis

Extract selected digits from the key as the address. Some digits of a telephone number or identification number may have good distinguishing power. The key characteristics must be understood first, and the extracted digits should be distributed reasonably evenly.

#### Mid-square method

Square the original key and use several middle digits of the result as the address. This method depends less on the original key distribution and is suitable when the keys are not very long and their distribution is difficult to analyze in advance.

#### Folding method

Split a key into several parts of equal width and add the parts to obtain an address. If the last part is shorter, it can be used as is. Folding from another direction can be added if one-way folding produces an uneven distribution.

#### Division-remainder method

Use the remainder after dividing the key by the table size $m$:

$$
h(key) = key \bmod m
$$

In practice, a prime table size near the desired size is often considered to reduce collisions caused by particular key distributions.

#### Random-number method

Use a pseudorandom function to generate an address from the key. The function must return the same result for the same key each time; otherwise, the record cannot be located again.

### Handling hash collisions

Once a collision is found, a collision-resolution method is needed to find another storage position.

#### Open addressing

When a collision occurs, continue searching for an empty slot according to a probe sequence:

$$
h_i(key) = (h(key)+d_i) \bmod m
$$

Different choices of $d_i$ produce different open-addressing methods:

- Linear probing: $d_i=i$. It is simple but prone to clustering.
- Quadratic probing: $d_i=\pm i^2$. It can reduce the clustering of linear probing.
- Random probing: $d_i$ comes from a reproducible pseudorandom sequence.

#### Rehashing

Use a second hash function to compute the probe step, for example:

$$
d_i=i\times h_2(key)
$$

When a collision occurs, the probe step changes until an empty slot is found or the table is confirmed to be full.

#### Separate chaining

Associate a linked list or another container with each hash address. A colliding record is placed in the container for that address; lookup first locates the address and then searches that container.

#### Common overflow area

Store colliding records in a separate overflow area. If a record is not found in the main table, search the overflow area as well. This is easy to implement when the number of collisions is small.

### Hash table search

The following example uses a linear-probing hash table with 16 slots. The slot record and the hash table are defined at the beginning; `None` represents an empty slot, so 0 does not need to be reserved as a special key.

```python
from dataclasses import dataclass


@dataclass
class HashEntry:
    """A record stored in one hash-table slot."""

    key: int
    value: object = None


class LinearProbingHashTable:
    """A key-value table using open addressing and linear probing."""

    def __init__(self, capacity: int = 16) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
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

Python lists are arrays, but the example uses `HashEntry` and `LinearProbingHashTable` to make the relationship between slots, probing, and key-value records explicit.

The example above uses integer keys. Strings and other data are eventually represented through encodings, so they can be converted to integers or handled with a language-provided hash function. Hash values and slot layouts should not be treated as a persistent data format.

## Sorting

Ordered data is useful in web search, report generation, and data analysis. Sorting rearranges the elements of a linear table so that their keys satisfy a non-increasing or non-decreasing relationship. The rest of this article assumes non-decreasing order.

### Basic sorting concepts and classifications

Multi-key sorting can be viewed as comparing several keys in sequence. In practice, multiple keys are often combined into one comparison rule; the algorithms below focus on a single key.

When two records have equal sorting keys, an algorithm is **stable** if their relative order is preserved; otherwise, it is **unstable**. For example, if two students have the same score and remain in their original order after sorting, the sort is stable.

Based on whether all data fits in memory, sorting can be divided into **internal sorting** and **external sorting**. This article focuses on internal sorting.

Internal sorting is commonly evaluated by time complexity, auxiliary space, and stability. Algorithmic complexity is a broader concept; time complexity is only one part of it.

By their main operations, sorting algorithms can be divided into insertion, exchange, selection, and merge sorts. In another common classification, bubble sort, simple selection sort, and direct insertion sort are considered simple sorts, while Shell sort, heap sort, merge sort, and quicksort are treated as improved sorts.

Sorting usually operates on a linear table. To make the exchange operation explicit, the examples below wrap Python lists in small custom array classes.

### Bubble sort

Bubble sort repeatedly compares adjacent records and swaps them when they are in the wrong order. Each pass moves the largest element in the unsorted portion to the end. It is simple, stable, and has $O(n^2)$ time complexity.

```python
from dataclasses import dataclass


@dataclass
class IntArray:
    """A mutable integer sequence with a shared swap operation."""

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

If a pass makes no swaps, the remaining portion is already ordered and the algorithm can stop early.

### Simple selection sort

Simple selection sort finds the minimum element in the unsorted portion on each pass and swaps it into the current starting position. Its comparison count is generally $O(n^2)$. It performs fewer swaps than bubble sort, but it is unstable.

```python
from dataclasses import dataclass


@dataclass
class SelectionArray:
    """A sortable integer sequence with an explicit swap operation."""

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

### Straight insertion sort

Straight insertion sort takes each unsorted element and inserts it into the already ordered prefix. It works well for small or nearly ordered sequences, has $O(n^2)$ time complexity, and is stable.

```python
from dataclasses import dataclass


@dataclass
class InsertionArray:
    """A mutable sequence that insertion sort modifies in place."""

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

### Shell sort

Shell sort improves direct insertion sort. It first chooses a gap, groups elements that are that distance apart, and insertion-sorts each group. The gap is then reduced until it becomes 1. The early passes move the sequence closer to order, so the final insertion sort performs fewer shifts.

Shell sort is unstable, and its time complexity depends on the gap sequence. It cannot be summarized accurately by one fixed expression. The example below uses the common `gap // 2` sequence.

```python
from dataclasses import dataclass


@dataclass
class ShellArray:
    """A mutable integer sequence used by Shell sort."""

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

### Heap sort

Heap sort improves simple selection sort. Selection sort repeatedly scans the unsorted portion; heap sort organizes those elements as a heap and takes the maximum or minimum directly from the root.

A heap is a complete binary tree with an ordering property. In a max-heap, every node is greater than or equal to its children; in a min-heap, every node is less than or equal to its children. When a complete binary tree is stored in a zero-based array, the children of `index` are `2 * index + 1` and `2 * index + 2`.

```python
from dataclasses import dataclass


@dataclass
class MaxHeap:
    """A max-heap backed by a list; size marks its active range."""

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

Heap sort has $O(n\log n)$ time complexity and $O(1)$ auxiliary space, but it is generally unstable.

### Merge sort

Merge sort repeatedly splits a sequence until each subsequence contains one element, then merges two ordered subsequences into a longer ordered sequence. The implementation below uses temporary storage for each merge.

```python
from dataclasses import dataclass


@dataclass
class MergeArray:
    """The integer sequence operated on by merge sort."""

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

Merge sort has $O(n\log n)$ time complexity and requires $O(n)$ auxiliary space. Using a less-than-or-equal comparison during merging preserves the order of equal elements, so this implementation is stable.

### Counting sort

Counting sort is not comparison-based. It uses input integers as indexes in a counting array, counts how often each value occurs, and reconstructs the result in index order. The input must therefore consist of integers whose key range is not much larger than the number of elements.

### Bucket sort

Bucket sort distributes elements into a finite number of buckets according to their value ranges, sorts each bucket, and then concatenates the buckets. The number of buckets and the distribution rule have a significant effect on performance; the method is easier to use effectively when the data is relatively uniform.

### Radix sort

Radix sort processes integers digit by digit instead of comparing complete keys. LSD (least significant digit) processing starts at the lowest digit, while MSD (most significant digit) processing starts at the highest. Each pass commonly uses stable counting sort or bucket sort as a subroutine.

- **MSD**: process from the highest digit, which is useful for partitioning by prefix.
- **LSD**: process from the lowest digit, and preserve stability on every pass.

### Quick sort

Quick sort selects a pivot and partitions the sequence into two parts: elements on the left are no greater than the pivot, and elements on the right are no smaller. It then recursively processes the two parts.

Quick sort is unstable. Its average time complexity is $O(n\log n)$, but if each pivot is close to the minimum or maximum value, the worst-case complexity degrades to $O(n^2)$.

The following implementation uses in-place partitioning. The `QuickArray` class at the beginning makes the mutable linear table explicit.

```python
from dataclasses import dataclass


@dataclass
class QuickArray:
    """The mutable sequence modified directly by quick sort."""

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

#### Choosing a pivot

Quick sort performance depends heavily on the pivot. Random selection and median-of-three selection can make partitions more balanced and reduce the chance of degeneration.

#### Small-array optimization

For very small subarrays, the overhead of further partitioning can exceed the work of sorting. Practical implementations can switch to straight insertion sort below a chosen size threshold.

#### Recursion optimization

Excessive recursion depth increases stack usage. One option is to recurse into the shorter partition first and process the longer partition in a loop, keeping additional stack space small.

## Conclusion

Quick sort is named for its strong performance in many practical situations, but it is not fastest for every input. Pivot selection, data distribution, stability requirements, and available memory all affect the choice of algorithm. Understanding these trade-offs is more useful than memorizing the name of one sorting algorithm.
