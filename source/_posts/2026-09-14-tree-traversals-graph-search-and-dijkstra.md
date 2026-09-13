---
title: "从二叉树遍历到图搜索：前中后序、层序与 DFS、BFS、Dijkstra"
title_en: "From Binary Tree Traversals to Graph Search: Preorder, Inorder, Postorder, Level Order, DFS, BFS, and Dijkstra"
date: 2026-09-14 12:00:00 +0800
categories: ["Programming", "CS Foundations"]
tags: ["Data Structures", "Algorithms"]
author: Hyacehila
excerpt: "把二叉树的前中后序与层序遍历放到图搜索中理解：DFS 何时处理节点，BFS 为什么能求最少边数，以及 Dijkstra 如何用累计距离选择下一个顶点。"
excerpt_en: "Connect binary tree traversals with graph search: when DFS processes a node, why BFS finds paths with the fewest edges, and how Dijkstra selects the next vertex by cumulative distance."
mathjax: false
permalink: '/blog/2026/09/14/tree-traversals-graph-search-and-dijkstra/'
---

二叉树的前序、中序、后序和层序，图的 DFS、BFS，再加上求最短路径的 Dijkstra 算是相当基础且常见的做法，这里希望整体结合起来复习一遍。

把它们放在一起看，会发现其中有两件可以分开考虑的事：下一步走向哪里，以及走到一个节点时，什么时候处理它。

前中后序沿着同一种深度优先的递归过程行走，区别在于处理节点的时机。层序遍历按层向外展开，对应广度优先搜索。到了带权图，Dijkstra 又把“下一步处理谁”的依据换成了从起点出发的累计距离。

给出二叉树。

```text
        A
       / \
      B   C
     / \   \
    D   E   F
```

| 遍历方式 | 处理顺序 | 这棵树的输出 |
| --- | --- | --- |
| 前序 | 根 → 左子树 → 右子树 | A B D E C F |
| 中序 | 左子树 → 根 → 右子树 | D B E A C F |
| 后序 | 左子树 → 右子树 → 根 | D E B F C A |
| 层序 | 从根开始，逐层从左向右 | A B C D E F |

“前、中、后”说的是根节点相对于左右子树的位置。这里每一棵子树都要完整地按同一规则处理，不能把“左子树”只理解成“左孩子”。

三种递归遍历可以写在同一段代码里。假设节点有 `value`、`left` 和 `right` 三个属性，空孩子用 `None` 表示：

```python
def traverse(node, pre, ino, post):
    if node is None:
        return
    pre.append(node.value)       # 刚进入当前节点
    traverse(node.left, pre, ino, post)
    ino.append(node.value)       # 左子树完成，右子树尚未开始
    traverse(node.right, pre, ino, post)
    post.append(node.value)      # 左右子树都已完成
```

传入三个空列表，一趟递归就能分别收集前序、中序和后序。递归调用的路线没有变化，改变的是把节点值记到结果里的位置。即使是后序遍历，程序也必须先进入根节点，才能找到它的孩子；最后输出根，不代表最后才到达根。

要理解这些位置的含义，可以把一次递归看成“负责处理一整棵子树”。站在 A 上，眼前的 B 代表包含 B、D、E 的整个子问题，C 代表包含 C、F 的另一个子问题。A 会把工作交给 B，等 B 全部完成后才继续，再把工作交给 C。每个节点内部都重复这个过程。

只看 A 这一层，过程是这样的：

```text
进入 A                         ← 前序位置
    进入 B，完成 B 的整棵子树
回到 A，准备处理另一侧          ← 中序位置
    进入 C，完成 C 的整棵子树
回到 A，结束 A 的整棵子树       ← 后序位置
```

这三个位置对应着不同的工作进度。前序时，当前节点已经拿到父节点传下来的信息，左右子问题还没开始；中序时，左子问题已经完成，右子问题还没开始；后序时，左右子问题都已完成，需要的话就可以使用它们返回的结果。

因此，选择遍历顺序时，可以先想清楚：处理当前节点需要什么信息，这些信息什么时候才能准备好？

前序适合那些到达节点时就能做的事，也适合先准备好孩子需要的信息。比如给树上的每个节点标注深度：根 A 的深度是 0，到了 B 就知道它的深度是 1，再把 2 传给 D、E。B 的深度来自它在祖先路径上的位置，计算时不需要知道 D、E 下面还有多少节点。信息沿着父子关系向下传递，前序位置就很自然。

后序适合当前答案依赖孩子答案的情况。假设要计算每个节点所代表的子树一共有多少节点。刚到 B 时，只知道 B 自己贡献 1，还不知道两棵子树各有多大。等 D 返回 1、E 返回 1，B 才能算出 `1 + 1 + 1 = 3`，再把 3 返回给 A。C 同样返回 2，最后 A 得到 `1 + 3 + 2 = 6`。答案从叶子开始，逐层向上汇总。

中序有一个更具体的用途：把当前节点放在左、右两部分之间，读出它在整体中的位置。“中”指的是这两部分之间，不代表遍历到了一半，也不代表当前节点两侧的节点数相等。

二叉搜索树正好赋予了这个位置大小关系。先考虑键互不相同的情形：某个节点的左子树里所有键都比它小，右子树里所有键都比它大。因此，先排好左边、再放当前键、最后排好右边，就能得到整棵树的升序结果。每棵子树内部继续遵守同样的规则，最终就得到完整的有序序列。如果允许重复键并采用一致的放置规则，结果是非递减序列。普通二叉树没有这个排序保证，但仍然可以按中序把左边、当前节点和右边依次读出来。

这些例子描述的是各个位置适合承担的工作。一个实际算法完全可以进入节点时记录路径，处理完左右子树后再汇总答案，在同一趟 DFS 里同时使用前序和后序位置。像前面的 `traverse` 一样，递归提供了几个处理时机，任务的信息依赖决定了每个时机应该做什么。

这三种遍历都属于 DFS（深度优先搜索）。程序沿一条分支向下走，完成子树后再返回，递归调用栈替我们记住了返回的位置。

层序遍历使用队列。先把根节点入队，每次取出队首并处理，再把它的非空左、右孩子依次放到队尾。处理 B 时，C 已经在队列里，所以新加入的 D、E 会排在 C 后面。先进先出就这样保证了上一层先于下一层处理。如果需要把结果分成一层一组，可以在每轮开始时记下队列长度，只处理这批节点。

下面的实现按层返回结果。节点仍使用前面约定的 `value`、`left`、`right` 三个属性，`None` 表示空孩子：

```python
from collections import deque

def level_order(root):
    if root is None:
        return []

    queue = deque([root])
    levels = []
    while queue:
        level_size = len(queue)  # 本轮开始时，队列里恰好是当前层
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.value)
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)
        levels.append(level)

    return levels
```

用一个简单的节点类建出文章开头的树，就可以直接运行：

```python
class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

root = TreeNode(
    "A",
    TreeNode("B", TreeNode("D"), TreeNode("E")),
    TreeNode("C", right=TreeNode("F")),
)
levels = level_order(root)
print(levels)                              # [['A'], ['B', 'C'], ['D', 'E', 'F']]
print([v for level in levels for v in level])  # ['A', 'B', 'C', 'D', 'E', 'F']
```

处理 B、C 这一层时，`level_size` 已经固定为 2。即使处理 B 后把 D、E 加入队列，这一轮也只会再取出 C；新加入的 D、E、F 留到下一轮。外层 `while` 推进层数，内层 `for` 处理当前层的节点。如果只需要一条连续的遍历序列，也可以每次出队就直接记录节点值，省去分层的内层循环。

从根沿孩子指针遍历一棵正常的树，每个非根节点都只有一个父节点，也没有环，不必额外记录是否来过。换成一般的图，同一个顶点可能从几条路径到达，也可能沿环走回原处，DFS 和 BFS 都需要防止重复搜索。

图的 DFS 可以在进入顶点时把它加入 `visited`，然后按邻接表顺序递归搜索尚未访问的邻居。图的 BFS 则在发现新顶点、准备入队时就标记它，避免几个顶点把同一个邻居反复加入队列。两者从一个起点出发都只覆盖可达部分；若要遍历整张图，还要遍历所有顶点，从尚未访问的顶点重新开始。

给前面的树加一条 E 与 F 之间的边，就得到下面这张无向图。B、A、C、F、E 之间形成了环，同一个顶点也有了多条到达路径，可以用它观察 `visited` 的作用。

```text
        A
       / \
      B   C
     / \   \
    D   E---F
```

用邻接表保存它，`graph[u]` 表示 u 的邻居。无向边在两端都要记录，例如 A 的列表里有 B，B 的列表里也有 A。下面固定按列表从左到右的顺序检查邻居：

```python
graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}
```

DFS 先用递归实现。`visit(u)` 一旦调用 `visit(v)`，就会暂停自己这一层的循环，等 v 那一层的搜索完成后再继续检查下一个邻居。

```python
def dfs(graph, start):
    visited = set()
    order = []

    def visit(u):
        visited.add(u)          # 进入时就标记，防止沿环再次进入
        order.append(u)         # 前序位置：记录首次进入的顺序
        for v in graph.get(u, []):
            if v not in visited:
                visit(v)        # 深入这一支，返回后再看下一个邻居
        # 后序位置：到这里，u 的所有邻居都已检查完

    visit(start)
    return order

print(dfs(graph, "A"))           # ['A', 'B', 'D', 'E', 'F', 'C']
```

从 A 进入 B 后，B 的第一个邻居 A 已经访问过，直接跳过。接着进入 D，D 没有未访问的邻居，于是返回 B。再从 B 进入 E，沿 E → F → C 继续深入。等这一支全部返回 A，A 才继续检查自己的邻居 C，此时 C 已经访问过了。

这里维持深度优先顺序的是递归调用栈。`visited` 在整个搜索中保留，递归返回时也不会删除顶点；这样 D 不会回到 B 重新搜索，F 也不会沿着环无限走下去。

若把 `order.append(u)` 移到 `for` 循环结束后，就会得到完成顺序 `D C F E B A`，对应图 DFS 的后序。移动记录位置时，入口处的 `visited.add(u)` 仍然要保留，这是实现 BFS 和 DFS 的习惯，一定要把标记放在最前面，避免重复查阅陷入死循环。

BFS 把等待处理的顶点放在队列里。下面同时记录遍历顺序、最少边数和前驱，便于和后面的最短路径讨论对应。`deque` 是 Python 的双端队列，这里只用队尾加入和队首取出。

```python
from collections import deque

def bfs(graph, start):
    visited = {start}            # 起点在入队时就标记
    queue = deque([start])
    order = []
    dist = {start: 0}
    parent = {start: None}

    while queue:
        u = queue.popleft()     # 取出等待最久的顶点
        order.append(u)
        for v in graph.get(u, []):
            if v in visited:
                continue
            visited.add(v)      # 发现时就标记，避免重复入队
            dist[v] = dist[u] + 1
            parent[v] = u
            queue.append(v)     # 放到队尾，等待后续处理

    return order, dist, parent

order, dist, parent = bfs(graph, "A")
print(order)                    # ['A', 'B', 'C', 'D', 'E', 'F']
print(dist)                     # {'A': 0, 'B': 1, 'C': 1, 'D': 2, 'E': 2, 'F': 2}
print(parent)                   # {'A': None, 'B': 'A', 'C': 'A', 'D': 'B', 'E': 'B', 'F': 'C'}
```

队首在左，处理完每个顶点后，队列的变化如下：

```text
开始       [A]
处理 A 后  [B, C]
处理 B 后  [C, D, E]
处理 C 后  [D, E, F]
处理 D 后  [E, F]
处理 E 后  [F]          F 已经入队并标记，不会再加入一次
处理 F 后  []
```

区别在 B 这一轮就能看出来：DFS 发现 D 后立刻进入 D，B 的循环暂时停下；BFS 发现 D 后只把它排到队尾，继续检查 E。等 B 处理完，队首是早已等着的 C，所以接下来处理 C。这也让 F 先通过 A → C → F 被发现，距离为 2。沿 `parent` 从 F 回溯得到 F、C、A，反转后就是这条路径。上面的 DFS 则先沿 A → B → E → F 到达 F，经过了 3 条边。

两段代码都假设起点是图中的顶点；没有出边的顶点可以省略邻接表条目，`graph.get(u, [])` 会把它当作空列表。返回结果只包含起点可达的顶点。

图没有天然的左孩子、右孩子，因此遍历结果还取决于邻居的枚举顺序。DFS 仍有进入顶点和完成顶点的时刻，对应前序与后序的思想；一般图没有统一的“左子树结束、右子树开始”位置，也就没有二叉树意义下的标准中序遍历。

BFS 在图上的用途还多了一层：它可以求无权图中从起点到各可达顶点的最少边数。起点距离为 0，它首次发现的邻居距离为 1，再往外是 2。队列始终按这个层次推进，所以一个顶点首次被发现时，已经不可能再从后面的层找到边数更少的路径。记录 `dist[v] = dist[u] + 1`，同时保存前驱 `parent[v] = u`，就能在搜索结束后反向还原一条最短路径。

DFS 没有这个保证。它可能先沿一条很深的分支走到目标，即使另一条路只需要两步。二叉树的层数与图中 BFS 的距离也在这里接上了：如果根的深度记作 0，那么节点所在的层，就是从根出发到它的边数。

一旦边有不同的权重，边数最少和总成本最低就可能分开。考虑下面这张有向图，数字表示边权：

```text
S ──10──→ A
│         ↑
1         1
↓         │
B ────────┘
```

S 到 A 有一条直接边，成本是 10；经由 B 需要两条边，总成本却只有 2。BFS 按边数会先发现直接到 A 的路径。如果我们关心总成本，就需要让成本为 1 的 B 先得到处理，再用经过 B 的路径更新 A。

Dijkstra 为每个顶点维护暂定距离 `dist`，表示目前已找到的路径中，从起点到它的最小成本。起点设为 0，其余设为无穷大。每轮从尚未确定最短距离的顶点里，选出 `dist` 最小的一个 u，再检查它的出边。如果经过 u 到 v 更便宜，就更新 `dist[v]`。这个尝试改进距离的操作叫松弛：

```text
如果 dist[u] + weight(u, v) < dist[v]：
    dist[v] = dist[u] + weight(u, v)
    parent[v] = u
```

选取最小暂定距离通常用最小堆，也就是优先队列。堆里的优先级是从起点到顶点的累计距离，不能只看刚经过的那条边有多小。

下面是允许同一顶点多次入堆的 Python 写法。`graph[u]` 保存 `(邻居, 边权)`，顶点使用字符串或整数编号，所有边权都必须非负。返回的字典只记录从起点可达的顶点。

```python
from heapq import heappop, heappush

def dijkstra(graph, start):
    dist = {start: 0}
    heap = [(0, start)]
    while heap:
        d, u = heappop(heap)
        if d != dist[u]:          # 更短的路径已经替换了这条旧记录
            continue
        for v, weight in graph.get(u, []):
            candidate = d + weight
            if candidate < dist.get(v, float("inf")):
                dist[v] = candidate
                heappush(heap, (candidate, v))
    return dist
```

在刚才的图里，处理 S 后，堆中有 `(1, B)` 和 `(10, A)`。先弹出 B，把 A 的距离改为 2，再加入 `(2, A)`。之后弹出 `(2, A)` 并处理它；旧的 `(10, A)` 留在堆里，轮到它时会因为距离已经过期而被跳过。

这也解释了为什么不能照搬 BFS 的标记方式：Dijkstra 首次发现 A 时只知道一条成本为 10 的路径，此时还不能把答案定下来。它要等 A 以当前最小的有效暂定距离出队时，才确定最短距离。若只求一个目标，也应该在这个时刻结束搜索。

Dijkstra 的这个判断依赖非负边权。当 u 的暂定距离已经是未确定顶点中最小的，任何绕经其他未确定顶点的路径，都无法靠后续的非负边把成本降到它以下。若存在负权边，后面的路径就可能把成本再拉低，这个论证便不成立。零权边可以使用，负权图则需要考虑 Bellman–Ford 等适用的算法。

| 方法 | 下一步优先处理谁 | 能保证什么 |
| --- | --- | --- |
| DFS | 当前分支上继续深入的未访问邻居，完成后回退 | 搜索可达顶点，不保证最短路径 |
| BFS | 最早入队的顶点，按距起点的边数推进 | 无权图的最少边数；各边等正权时也是最小总权重 |
| Dijkstra | 当前累计暂定距离最小的未确定顶点 | 非负权图的单源最短路径 |

所有边权都为 1 时，Dijkstra 按累计距离推进的次序就与 BFS 的分层一致，同层内部的顺序可以不同。此时普通队列已经足够。对于一般的非负权图，则需要同时改变选点顺序和距离更新方式，仅仅把 BFS 的队列替换成堆还不够。

从开销看，含 n 个节点的二叉树，四种遍历的时间都是 O(n)。不计结果列表，递归 DFS 的额外空间是 O(h)，h 为树高；层序遍历是 O(w)，w 为最大层宽。很深的树会占用更多递归栈空间，很宽的树会占用更多队列空间。

用邻接表存图，DFS 和 BFS 的时间都是 O(V + E)，额外空间为 O(V)，V、E 分别是顶点数和边数。常见的二叉堆 Dijkstra 在简单图上可写作 O((V + E) log V)。上面的重复入堆版本最多产生 O(E) 条堆记录；若允许大量平行边，更准确的时间上界是 O(V + E log(E + 1))，额外空间为 O(V + E)。这些开销都不计输入图本身。

实际写题时，可以先确认需要什么结果。要把子树的信息汇总到父节点，处理逻辑就放在后序位置；要找最少走几条边，用 BFS；要让非负边权的总和最小，用 Dijkstra。至于图有没有环、顶点何时标记、路径是否还会变短，则决定了那些看起来很像的代码，哪些部分可以复用，哪些必须重新考虑。
