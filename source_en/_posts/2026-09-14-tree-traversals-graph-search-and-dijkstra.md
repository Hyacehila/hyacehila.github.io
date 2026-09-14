---
title: "From Binary Tree Traversals to Graph Search: Preorder, Inorder, Postorder, Level Order, DFS, BFS, and Dijkstra"
title_zh: "从二叉树遍历到图搜索：前中后序、层序与 DFS、BFS、Dijkstra"
date: 2026-09-14 12:00:00 +0800
categories: ["Programming", "CS Foundations"]
tags: ["Data Structures", "Algorithms"]
author: Hyacehila
excerpt: "Connect binary tree traversals with graph search: when DFS processes a node, why BFS finds paths with the fewest edges, and how Dijkstra selects the next vertex by cumulative distance."
description: "Connect binary tree traversals with graph search: when DFS processes a node, why BFS finds paths with the fewest edges, and how Dijkstra selects the next vertex by cumulative distance."
excerpt_zh: "把二叉树的前中后序与层序遍历放到图搜索中理解：DFS 何时处理节点，BFS 为什么能求最少边数，以及 Dijkstra 如何用累计距离选择下一个顶点。"
mathjax: false
hidden: true
permalink: '/blog/2026/09/14/tree-traversals-graph-search-and-dijkstra/'
lang: en
translation_key: 2026-09-14-tree-traversals-graph-search-and-dijkstra
translation_status: machine
translation_source_hash: ac4947fc4b3181f9ae08f82a3c9d712f3d5d3636cf60d0bd0523f27559847e32
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

Preorder, inorder, postorder, and level-order traversal of binary trees, DFS and BFS on graphs, and Dijkstra's shortest-path algorithm are all familiar fundamentals. I want to review them together here.

Looking at them together reveals two questions that we can consider separately: where to go next, and when to process a node once we reach it.

Preorder, inorder, and postorder follow the same depth-first recursive process; they differ in when they process each node. Level-order traversal expands outward one level at a time, corresponding to breadth-first search. On a weighted graph, Dijkstra changes the basis for choosing the next vertex to the cumulative distance from the starting point.

Consider this binary tree:

```text
        A
       / \
      B   C
     / \   \
    D   E   F
```

| Traversal | Processing order | Output for this tree |
| --- | --- | --- |
| Preorder | Root → left subtree → right subtree | A B D E C F |
| Inorder | Left subtree → root → right subtree | D B E A C F |
| Postorder | Left subtree → right subtree → root | D E B F C A |
| Level order | Starting at the root, level by level from left to right | A B C D E F |

The names preorder, inorder, and postorder describe the root's position relative to its left and right subtrees. Each subtree must be processed in full using the same rule. The left subtree means more than just the left child.

We can express all three recursive traversals in one piece of code. Assume each node has `value`, `left`, and `right` attributes, with `None` representing a missing child:

```python
def traverse(node, pre, ino, post):
    if node is None:
        return
    pre.append(node.value)       # Just entered the current node
    traverse(node.left, pre, ino, post)
    ino.append(node.value)       # Left subtree finished; right subtree not started
    traverse(node.right, pre, ino, post)
    post.append(node.value)      # Both subtrees finished
```

Pass in three empty lists, and one recursive traversal collects the preorder, inorder, and postorder sequences separately. The route taken by the recursive calls stays the same; what changes is where we record each node's value. Even in postorder, the program must enter the root before it can find the children. Outputting the root last does not mean reaching it last.

To understand these positions, think of each recursive call as taking responsibility for an entire subtree. From A, B represents the whole subproblem containing B, D, and E; C represents another subproblem containing C and F. A hands work to B, waits for B to finish completely, and then hands work to C. Every node repeats this process internally.

Looking only at A's call, the process is:

```text
Enter A                              ← Preorder position
    Enter B and finish B's entire subtree
Return to A, ready for the other side ← Inorder position
    Enter C and finish C's entire subtree
Return to A and finish A's subtree    ← Postorder position
```

These positions represent different stages of progress. At the preorder position, the current node has received information passed down by its parent, but neither child subproblem has started. At the inorder position, the left subproblem is finished and the right one has not started. At the postorder position, both subproblems are finished, so their returned results are available if needed.

When choosing a traversal order, first consider what information is needed to process the current node, and when that information will be ready.

Preorder suits work that can be done upon reaching a node, as well as preparing information its children will need. For example, to label every node with its depth, the root A has depth 0. Upon reaching B, we know its depth is 1, and we can pass 2 to D and E. B's depth comes from its position along the path through its ancestors; we do not need to know how many nodes lie below D or E. Information flows down the parent-child relationships, so the preorder position is a natural fit.

Postorder suits cases where the current answer depends on the children's answers. Suppose we want the number of nodes in each node's subtree. Upon reaching B, we know B itself contributes 1, but we do not yet know the sizes of its two subtrees. Once D returns 1 and E returns 1, B can compute `1 + 1 + 1 = 3` and return 3 to A. C similarly returns 2, and A finally obtains `1 + 3 + 2 = 6`. Results start at the leaves and are combined upward, one level at a time.

Inorder has a more specific use: placing the current node between its left and right parts to read its position within the whole. The middle here means between those two parts. It does not mean that traversal is halfway finished or that the two sides contain equal numbers of nodes.

A binary search tree gives that position an ordering by key value. First consider distinct keys: every key in a node's left subtree is smaller than its key, and every key in its right subtree is larger. Listing the left side in order, then the current key, and finally the right side in order produces an ascending sequence for the whole tree. Applying the same rule within every subtree produces the complete sorted sequence. If duplicate keys are allowed and placed consistently, the result is nondecreasing. An ordinary binary tree has no such sorting guarantee, although inorder can still read its left part, current node, and right part in sequence.

These examples describe the work suited to each position. An actual algorithm can record the current path upon entering a node and combine results after processing both subtrees, using preorder and postorder positions within the same DFS. As in `traverse` above, recursion provides several opportunities to do work; the task's information dependencies determine what belongs at each one.

All three traversals are forms of DFS, or depth-first search. The program follows a branch downward and returns after completing a subtree. The recursive call stack remembers where to return.

Level-order traversal uses a queue. Enqueue the root first. Each time, remove and process the node at the front, then append its nonempty left and right children to the back, in that order. When B is processed, C is already in the queue, so the newly added D and E go behind C. This first-in, first-out rule ensures that one level is processed before the next. To group the result by level, record the queue's length at the start of each round and process only that batch of nodes.

The following implementation returns results grouped by level. Nodes still use the `value`, `left`, and `right` attributes introduced earlier, with `None` representing a missing child:

```python
from collections import deque

def level_order(root):
    if root is None:
        return []

    queue = deque([root])
    levels = []
    while queue:
        level_size = len(queue)  # At the start of this round, the queue holds this level
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

Use a simple node class to construct the tree from the beginning of the article, and the example is ready to run:

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

When processing the level containing B and C, `level_size` is fixed at 2. Even though processing B adds D and E to the queue, this round removes only C next. The newly added D, E, and F wait for the next round. The outer `while` advances through levels, while the inner `for` processes the nodes in the current level. If all you need is one flat traversal sequence, you can record each node's value as it leaves the queue and omit the inner loop that groups nodes by level.

When traversing a normal tree from the root through child pointers, every nonroot node has exactly one parent and there are no cycles, so there is no need to track whether a node has been visited. In a general graph, the same vertex may be reachable by several paths, and a cycle may lead back to an earlier vertex. Both DFS and BFS need to prevent repeated searches.

Graph DFS can add a vertex to `visited` upon entering it, then recursively search unvisited neighbors in adjacency-list order. Graph BFS marks a new vertex when it is discovered, just before enqueueing it, so several vertices cannot repeatedly add the same neighbor to the queue. Starting from one vertex, both algorithms cover only the reachable part of the graph. To traverse the entire graph, iterate over all vertices and restart from any vertex that remains unvisited.

Adding an edge between E and F to the earlier tree produces the following undirected graph. B, A, C, F, and E form a cycle, and vertices now have multiple paths leading to them. This lets us observe the role of `visited`.

```text
        A
       / \
      B   C
     / \   \
    D   E---F
```

Store it as an adjacency list, where `graph[u]` contains u's neighbors. An undirected edge must be recorded at both ends: for example, A's list contains B, and B's list contains A. Below, neighbors are always checked from left to right in each list:

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

We first implement DFS recursively. Once `visit(u)` calls `visit(v)`, the loop in u's call pauses until the search in v's call finishes, then resumes checking the next neighbor.

```python
def dfs(graph, start):
    visited = set()
    order = []

    def visit(u):
        visited.add(u)          # Mark on entry to prevent reentering through a cycle
        order.append(u)         # Preorder position: record the order of first entry
        for v in graph.get(u, []):
            if v not in visited:
                visit(v)        # Explore this branch before checking the next neighbor
        # Postorder position: all of u's neighbors have now been checked

    visit(start)
    return order

print(dfs(graph, "A"))           # ['A', 'B', 'D', 'E', 'F', 'C']
```

After entering B from A, B's first neighbor A has already been visited and is skipped. The search then enters D. D has no unvisited neighbors, so it returns to B. From B, the search next enters E and continues along E → F → C. Only after this whole branch returns to A does A continue checking its own neighbor C, which has already been visited by then.

The recursive call stack maintains the depth-first order here. `visited` is retained throughout the search, and vertices are not removed when recursive calls return. This prevents D from searching B again and F from continuing around the cycle indefinitely.

Moving `order.append(u)` to after the `for` loop produces the finishing order `D C F E B A`, corresponding to postorder in graph DFS. When moving the recording step, keep `visited.add(u)` at the entry point. Marking early is the convention used in these BFS and DFS implementations to prevent repeated searches from getting trapped in a cycle.

BFS puts vertices awaiting processing into a queue. The following code records traversal order, minimum edge counts, and predecessors, connecting the implementation to the shortest-path discussion that follows. Python's `deque` is a double-ended queue; here we only append at the back and remove from the front.

```python
from collections import deque

def bfs(graph, start):
    visited = {start}            # Mark the starting vertex when it is enqueued
    queue = deque([start])
    order = []
    dist = {start: 0}
    parent = {start: None}

    while queue:
        u = queue.popleft()     # Remove the vertex that has waited longest
        order.append(u)
        for v in graph.get(u, []):
            if v in visited:
                continue
            visited.add(v)      # Mark on discovery to prevent duplicate enqueueing
            dist[v] = dist[u] + 1
            parent[v] = u
            queue.append(v)     # Append at the back for later processing

    return order, dist, parent

order, dist, parent = bfs(graph, "A")
print(order)                    # ['A', 'B', 'C', 'D', 'E', 'F']
print(dist)                     # {'A': 0, 'B': 1, 'C': 1, 'D': 2, 'E': 2, 'F': 2}
print(parent)                   # {'A': None, 'B': 'A', 'C': 'A', 'D': 'B', 'E': 'B', 'F': 'C'}
```

With the front on the left, the queue changes as follows after each vertex is processed:

```text
Start    [A]
After A  [B, C]
After B  [C, D, E]
After C  [D, E, F]
After D  [E, F]
After E  [F]          F is already enqueued and marked; it is not added again
After F  []
```

The difference is already visible when processing B. DFS immediately enters D upon discovering it, pausing B's loop. BFS only appends D to the back of the queue and continues checking E. Once B is finished, C is at the front, having already been waiting, so C is processed next. As a result, F is first discovered through A → C → F, at distance 2. Following `parent` backward from F yields F, C, A; reversing that sequence gives the path. The DFS above first reaches F along A → B → E → F, using 3 edges.

Both implementations assume the starting point is a vertex in the graph. Vertices with no outgoing edges may be omitted from the adjacency-list dictionary; `graph.get(u, [])` treats them as having empty neighbor lists. The returned results contain only vertices reachable from the start.

A graph has no inherent left or right child, so traversal results also depend on the order in which neighbors are enumerated. DFS still has entry and finishing times, corresponding to the ideas of preorder and postorder. A general graph has no standard point where the left subtree ends and the right subtree begins, so it has no standard inorder traversal in the binary-tree sense.

BFS has another use on graphs: it finds the minimum number of edges from the starting vertex to every reachable vertex in an unweighted graph. The start has distance 0, its newly discovered neighbors have distance 1, and the next layer has distance 2. The queue always advances in this order, so when a vertex is first discovered, a path with fewer edges cannot later appear from a subsequent layer. Record `dist[v] = dist[u] + 1` along with the predecessor `parent[v] = u`, and after the search you can reconstruct a shortest path by tracing backward.

DFS offers no such guarantee. It may reach the target along a deep branch first, even when another path requires only two steps. This also connects binary-tree levels with BFS distances in graphs: if the root's depth is 0, a node's level is the number of edges from the root to that node.

Once edges have different weights, the path with the fewest edges may differ from the path with the lowest total cost. Consider this directed graph, where the numbers are edge weights:

```text
S ──10──→ A
│         ↑
1         1
↓         │
B ────────┘
```

The direct edge from S to A costs 10. Going through B uses two edges but costs only 2 in total. BFS, working by edge count, discovers the direct path to A first. If total cost matters, we need to process B, whose cost is 1, first and then update A using the path through B.

Dijkstra maintains a tentative distance `dist` for every vertex: the lowest cost from the start among the paths found so far. Set the starting distance to 0 and all others to infinity. In each round, select the vertex u with the smallest `dist` among those whose shortest distances have not yet been finalized, then examine its outgoing edges. If reaching v through u is cheaper, update `dist[v]`. This attempt to improve a distance is called relaxation:

```text
If dist[u] + weight(u, v) < dist[v]:
    dist[v] = dist[u] + weight(u, v)
    parent[v] = u
```

A min-heap, or priority queue, is commonly used to select the smallest tentative distance. The priority is the cumulative distance from the starting point to the vertex, not just the weight of the most recently traversed edge.

Here is a Python implementation that allows a vertex to be inserted into the heap more than once. `graph[u]` contains `(neighbor, weight)` pairs, vertices use string or integer identifiers, and all edge weights must be nonnegative. The returned dictionary records only vertices reachable from the start.

```python
from heapq import heappop, heappush

def dijkstra(graph, start):
    dist = {start: 0}
    heap = [(0, start)]
    while heap:
        d, u = heappop(heap)
        if d != dist[u]:          # A shorter path has superseded this old entry
            continue
        for v, weight in graph.get(u, []):
            candidate = d + weight
            if candidate < dist.get(v, float("inf")):
                dist[v] = candidate
                heappush(heap, (candidate, v))
    return dist
```

In the graph above, processing S leaves `(1, B)` and `(10, A)` in the heap. B is removed first, A's distance is changed to 2, and `(2, A)` is inserted. Next, `(2, A)` is removed and processed. The old `(10, A)` entry remains in the heap and is skipped when its turn comes because its distance is stale.

This explains why we cannot copy BFS's marking rule directly: when Dijkstra first discovers A, it knows only a path costing 10 and cannot finalize the answer yet. A's shortest distance is finalized when A is removed with the smallest current valid tentative distance. If searching for just one target, that is also the point at which the search should stop.

This reasoning depends on nonnegative edge weights. Once u has the smallest tentative distance among unfinalized vertices, a detour through another unfinalized vertex cannot use subsequent nonnegative edges to lower the cost below it. With negative edges, a later path could reduce the cost further, so the argument no longer holds. Zero-weight edges are allowed; for graphs with negative weights, consider an applicable algorithm such as Bellman–Ford.

| Method | Which vertex is processed next? | What does it guarantee? |
| --- | --- | --- |
| DFS | An unvisited neighbor deeper along the current branch, backtracking when finished | Searches reachable vertices; does not guarantee shortest paths |
| BFS | The vertex enqueued earliest, advancing by edge count from the start | Minimum edge counts in unweighted graphs; also minimum total weight when all edges have the same positive weight |
| Dijkstra | The unfinalized vertex with the smallest current tentative cumulative distance | Single-source shortest paths in graphs with nonnegative weights |

When every edge has weight 1, Dijkstra's progression by cumulative distance follows the same layers as BFS, though the order within a layer may differ. A regular queue is sufficient in this case. For general nonnegative weighted graphs, both the vertex-selection order and the distance-update rules must change. Simply replacing BFS's queue with a heap is not enough.

For a binary tree with n nodes, all four traversals take O(n) time. Excluding the result lists, recursive DFS uses O(h) extra space, where h is the tree height, while level-order traversal uses O(w), where w is the maximum width of a level. A deep tree consumes more recursive stack space, and a wide tree consumes more queue space.

With adjacency lists, DFS and BFS both take O(V + E) time and O(V) extra space, where V and E are the numbers of vertices and edges. The usual binary-heap implementation of Dijkstra can be bounded by O((V + E) log V) on a simple graph. The version above with repeated heap insertions creates at most O(E) heap entries; if arbitrarily many parallel edges are allowed, a more precise time bound is O(V + E log(E + 1)), with O(V + E) extra space. These space bounds exclude the input graph itself.

When solving a problem, first identify the result you need. To combine subtree information at a parent, place the work at the postorder position. To find the fewest edges, use BFS. To minimize the sum of nonnegative edge weights, use Dijkstra. Whether the graph has cycles, when a vertex should be marked, and whether a path can still become shorter determine which parts of these similar-looking implementations can be reused and which need to be reconsidered.
