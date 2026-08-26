---
title: 'Search and Sorting Algorithms: Binary Search, Hash Tables, and Sorting Implementations'
title_zh: 查找与排序算法：二分查找、散列表与排序实现
date: 2024-12-18 21:37:43 +0800
categories:
- Programming
- Computer Science Fundamentals
tags:
- Algorithms
- Search
- Sorting
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers sequential search, binary search, binary search trees, hash tables, sorting algorithms, and related C implementations.
description: Covers sequential search, binary search, binary search trees, hash tables, sorting algorithms, and related C
  implementations.
excerpt_zh: 整理顺序查找、二分查找、二叉排序树、散列表、排序算法和相关 C 语言实现。
permalink: /blog/2024/12/18/search-and-sorting-algorithms/
lang: en
translation_key: 2024-12-18-search-and-sorting-algorithms
translation_status: machine
translation_source_hash: 4c4cbb8fb7b92c401c070cadc6573a926de28eb4d122ef2f6192d0d28ec33105
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Find</h2>
<p>The questions in this article can also be addressed<a href="/en/blog/2025/05/12/data-structures-introduction/">Data structure introduction: linear tables, trees, maps and search sequence</a>、<a href="/en/blog/2025/05/13/algorithm-design-and-analysis/">Algorithm design and analysis: partitioning, dynamic planning and algorithms</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>It's a very big problem, and Search is a tool that any programmer must know and use, and it's at the heart of modern science, and it's possible that some of the things that are special about finding tools are the ones that you need to learn.</p>
<h3>Find General</h3>
<p>All data that need to be checked are called the collections of data<strong>Search Table</strong> Search Table</p>
<p><strong>Keywords</strong> Key is a certain item in the data element that he can identify a data element or a field (key code)</p>
<p>If the keyword is the only one that can identify a record, it's called him. <strong>Main keyword</strong> Primary Key, otherwise called subkey.</p>
<p>So the search is based on a value that we've found.</p>
<p>We return to position when we have a successful search or we return to empty pointer (normal)</p>
<p>By way of operation, we split the search tables into static and dynamic search tables.</p>
<p><strong>Static Search Sheet</strong> We're looking for the kind of search we want in a bunch of data.</p>
<p><strong>Dynamic Search Table</strong> Data elements are inserted or removed during search  </p>
<p>To improve the efficiency of the search, we often set the right data structure for the search operation. Yes.</p>
<p>The right structure is the general pattern of the discussion that we're going to discuss, the static search of the linear table, the dynamic search of the dident sorting tree, and the selection of the scattered list for some special cases.</p>
<h3>Order Table Searches</h3>
<h4>Order Searches</h4>
<p>At this point, the data elements are a linear table of disorder, and we have no good technique, and one by one, the only way to do it is to compare it.</p>
<pre><code class="language-c">int linear_search(int arr[N], int value) {
    int i;
    for (i = 0; i &lt; N; i++) {
        if (arr[i] == value) {
            return i;
        }
    }
    return 0;
}
//优化代码 避免一次越界检测
int linear_search(int arr[N], int value) {
    int i;
    a[0]=key;
    i=n;
    while(a[i]!=key){
        i--;
    }
    return i;
}
</code></pre>
<h4>Order Table Search</h4>
<p>If the original elements are in a certain order, it'll be very helpful to find out.</p>
<h4>Half Find</h4>
<pre><code class="language-c">int search(int nums[], int size, int target) //nums是数组，size是数组的大小，target是需要查找的值
{
    int left = 0;
    int right = size - 1;	// 定义了target在左闭右闭的区间内，[left, right]
    while (left &lt;= right) {	//当left == right时，区间[left, right]仍然有效
        int middle = left + ((right - left) / 2);//等同于 (left + right) / 2 /符号自动取整
        if (nums[middle] &gt; target) {
            right = middle - 1;	//target在左区间，所以[left, middle - 1]
        } else if (nums[middle] &lt; target) {
            left = middle + 1;	//target在右区间，所以[middle + 1, right]
        } else {	//既不在左边，也不在右边，那就是找到答案了
            return middle;
        }
    }
    return -1;
}
//原理非常容易 处理好边界就可以了 不复杂
</code></pre>
<h4>Plugin Search</h4>
<p>Can we start at half the place? Can't we just start somewhere else? Mathematicians gave their answers, plugged in the formula.</p>
<pre><code class="language-c">mid = low + (high-low)*(key-a[low])/(a[high]-a[low]);
</code></pre>
<p>Using plug-in formulas, sometimes it increases the efficiency of searching, for example, when arrays are long and even.</p>
<h4>Fabonacci.</h4>
<p>And it started from the partition point because F(n)=F(n-1)+F(n-2), so we chose to divide the array into two parts.</p>
<pre><code class="language-c">int Fibonacci_Search(int *a, int key, int n)
{
	int i, low = 0, high = n - 1;
	int mid = 0;
	int k = 0;
	int F[ARRSIZE];
	InitFibonacci(F);
	while (n &gt; F[k] - 1)          //计算出n在斐波那契中的数列
	{
		++k;
	}
	for (i = n; i &lt; F[k] - 1; ++i) //把数组补全  最大补全到后面的位置
	{
		a[i] = a[high];
	}
	while (low &lt;= high)
	{
		mid = low + F[k - 1] - 1;  //根据斐波那契数列进行黄金分割
		if (a[mid] &gt; key)
		{
			high = mid - 1;
			k = k - 1;
		}
		else if (a[mid] &lt; key)
		{
			low = mid + 1;
			k = k - 2;
		}
		else
		{
			if (mid &lt;= high) //如果为真则找到相应的位置
			{
				return mid;
			}
			else
			{
				return n;
			}
		}
	}
	return 0;
}
</code></pre>
<h3>Linear Index Search</h3>
<p>If the data is huge, it's impossible to get them in order, but can we get a little bit of a pattern in the order? </p>
<p>Index structure is generally divided into <strong>Linear Index Tree Index Multi-level Index</strong> We're here to highlight the linear index structure.</p>
<h4>Thin Index</h4>
<p>The record in each data set corresponds to a single index item, which must be organized in a dense index by key code.</p>
<h4>Part Index</h4>
<p>The library's books are essentially a block index, and we're sort of sort of sort of sort of sort of sort of sort of sort of sort of sort of sort of sort of thing, sort of sort of sort of sort of sort of thing, sort of sort of sort of thing, because he's got a guy who's got a lot of work on him and he's not really a man.</p>
<h4>Backward Index</h4>
<p>It's actually a more common technique for search engines, and it's easy to find that the efficiency of search engines is as high as it is scary.</p>
<p>We take all the key words that may be individually and find their original file number, and then when we enter the key word, it's very efficient to search because we can sort the key words out or create index structures.</p>
<p>This indexing technique is called a reverse index, which is based on a key code and log number. </p>
<h3>A tree sorted fork</h3>
<p>Let's try it now. <strong>Dynamic Search Table</strong>    We want to find a good search and on the one hand, we want to be able to insert and delete it, as the title of this chapter is, we want to use the fork tree to achieve a comparison of the size of one layer of the new element with the elements in the original tree, and we want to put the new element on the left side, and we want to have one layer on the right side, and then we want to have one layer of the last one, and this is an orderly sequence of the two-fork tree, which is the binary soort Tree.</p>
<ul>
<li>If the left tree is not empty, all the points of the left tree are less than the value of his structure.</li>
<li>If the right subtree is not empty, all the nodes of the right subtree have values greater than the root node.</li>
<li>The right and right sub-trees are sorted with two fork trees.</li>
</ul>
<p>The retrogression is the subject of a two-knot tree that we can't miss, not to sort, but to find, insert and delete, and then some of the codes are there.</p>
<pre><code class="language-c">typedef int DataType;
typedef struct BST_Node {
    DataType data;
    struct BST_Node *lchild, *rchild;
}BST_T, *BST_P;
//我们先默认有一棵二叉排序树 建立放到插入后面讲 很快就会理解意思
BST_P SearchMin(BST_P root)
{
    if (root == NULL)
        return NULL;
    if (root-&gt;lchild == NULL)
        return root;
    else  //一直往左孩子找，直到没有左孩子的结点
        return SearchMin(root-&gt;lchild);
}
//查找最大原理也非常的简单 略去了
BST_P Search_BST(BST_P root, DataType key)
{
    if (root == NULL)
        return NULL;
    if (key &gt; root-&gt;data) //查找右子树
        return Search_BST(root-&gt;rchild, key);
    else if (key &lt; root-&gt;data) //查找左子树
        return Search_BST(root-&gt;lchild, key);
    else
        return root;
}
//递归是查找算法的核心 学习这些也会帮助我们理解递归
void Insert_BST(BST_P *root, DataType data)
{
    //初始化插入节点
    BST_P p = (BST_P)malloc(sizeof(struct BST_Node));
    if (!p) return;
    p-&gt;data = data;
    p-&gt;lchild = p-&gt;rchild = NULL;

    //空树时，直接作为根节点
    if (*root == NULL)
    {
        *root = p;
        return;
    }

    //是否存在，已存在则返回，不插入
    if (Search_BST(root, data) != NULL) return;

    //进行插入，首先找到要插入的位置的父节点
    BST_P tnode = NULL, troot = *root;
    while (troot)
    {
        tnode = troot;
        if(data &lt; troot-&gt;data){
            troot = troot-&gt;lchild;
        }
        else{
            troot = troot-&gt;rchild；
        }
    }
    if (data &lt; tnode-&gt;data)
        tnode-&gt;lchild = p;
    else
        tnode-&gt;rchild = p;
}
//所谓的插入就是查找一个合适的地方添加进去
void CreateBST(BST_P *T, int a[], int n)
{
    int i;
    for (i = 0; i &lt; n; i++)
    {
        Insert_BST(T, a[i]);
    }
}
//所谓的建立就是重复插入的过程 非常简单
void DeleteBSTNode(BST_P *root, DataType data)
{
    BST_P p = *root, parent = NULL, s = NULL;

    if (!p) return;

    if (p-&gt;data == data) //找到要删除的节点了
    {
        /* It&#39;s a leaf node */
        if (!p-&gt;rchild &amp;&amp; !p-&gt;lchild)
            *root = NULL;

        // 只有一个左节点
        else if (!p-&gt;rchild&amp;&amp;p-&gt;lchild)
            *root = p-&gt;lchild;

        // 只有一个右节点
        else if (!p-&gt;lchild&amp;&amp;p-&gt;rchild)
            *root = p-&gt;rchild;

        //左右节点都不空 里面是一个复杂的判断过程
        else
        {
            s = p-&gt;rchild;
            /* the s without left child */
            if (!s-&gt;lchild)
                s-&gt;lchild = p-&gt;lchild;  //完成接树的过程

            /* the s have left child */
            else
            {
                /* find the smallest node in the left subtree of s */
                while (s-&gt;lchild)
                {
                    /* record the parent node of s */
                    parent = s;
                    s = s-&gt;lchild;
                }
                parent-&gt;lchild = s-&gt;rchild;
                s-&gt;lchild = p-&gt;lchild;
                s-&gt;rchild = p-&gt;rchild;
            }
            *root = s;
        }
        free(p);
    }
    else if (data &gt; p-&gt;data) //向右找
        DeleteBSTNode(&amp;(p-&gt;rchild), data);
    else if (data &lt; p-&gt;data) //向左找
        DeleteBSTNode(&amp;(p-&gt;lchild), data);
}
//删除其实比较复杂 你得分清是不是叶子结点 如果要删除的结点有子孙 我们需要怎么变化我们的二叉树
//删除代码包括了我们进行查找的过程 后面的递归就是如此
</code></pre>
<p>The main part of the dident tree is these, and we can easily find a problem, and we want the tree to be more balanced, deep and fully balanced, so that the number of judgments that can be found for consumption is reduced, that is, the problem of balancing the dident tree.</p>
<h3>Balance the fork tree AVL tree</h3>
<p>The simple point of balance is that we want to see a maximum difference between the height of all the node left and the right tree, and we call the value of the diagonal node left tree minus the depth of the right subtree as a balance factor, and BF, for the AVL tree balance factor, only 10-1, and of course, if it's not sorted two fork tree, then there's no precondition for balancing two fork tree. </p>
<p>We call the tree closest to the plug and the fraction is the one whose absolute value is more than one. </p>
<p>The core of the construction of the balanced diagonal tree is to check whether the balance still exists when a node is inserted in the construction of the diagonal sort of tree, and if the balance is compromised, then adjust the logic between the fewer and lesser fractions of the tree, while maintaining the pre-ordering properties of the dident, so that he can be called the new balance tree, and then we can give you a thought, and let's just say, a simple example,</p>
<p>And how do you achieve this transformation, as the theoretically optimized tree of the didental sorting tree of the direct composition of the two fork, as shown in the figure?</p>
<p>Let's start from the beginning, and think about when there's an imbalance, and what should we do about it? </p>
<p>One, after inserting this fork tree, we found that the whole tree became the smallest imbalance subtree, and in order to balance it, we needed a whole clockwise rotation to make two roots. Points </p>
<p>When normal four inserts five, the smallest imbalance in node three subtrees needs to rotate reverse clockwise. </p>
<p>Add node six, and the whole tree becomes once again the smallest imbalance subtree, rotating the two-point counterclockwise.</p>
<p>By the same token, seven plus 10 plus nine times, but then it's not gonna solve the problem by going straight to the point where the BF of the smallest imbalance tree and his BF symbol of the subtree are going to have to do twice.</p>
<p>It's four situations, and it's not hard to understand.</p>
<p>The idea of this rotation is the idea of the AVL tree, and the way it is inserted, is to solve it.</p>
<pre><code class="language-c">struct node {
    int             data;
    int             height;
    struct node     *left;
    struct node     *right;
}

typedef struct node node_t;
typedef struct node* nodeptr_t;
//首先重新纠正我们的结点问题 没有高度怎么考虑平衡因子的问题呢
int treeHeight(nodeptr_t root) {
    if(root == NULL) {
        return -1;
    } else {
        return max(treeHeight(root-&gt;left),treeHeight(root-&gt;right)) + 1;
    }
}
//用来获得结点高度的函数 在后面进行什么删除或者插入的操作的时候记得更新高度这个量
int treeGetBalanceFactor(nodeptr_t root) {
    if(root == NULL)
        return 0;
    else
        return x-&gt;left-&gt;height - x-&gt;right-&gt;height;
}
//检测BF因子 当绝对值大于一的时候就应该进行一次修正
nodeptr_t treeRotateRight(nodeptr_t root) {
    nodeptr_t left = root-&gt;left; //保存新的根节点 也就是原本结点的左孩子
    root-&gt;left = left-&gt;right; // 将将要被抛弃的节点连接为旋转后的 root 的左孩子
    left-&gt;right = root; // 调换父子关系

    left-&gt;height = max(treeHeight(left-&gt;left), treeHeight(left-&gt;right))+1;
    right-&gt;height = max(treeHeight(right-&gt;left), treeHeight(right-&gt;right))+1;

    return left; //返回的是这一部分的新的根节点
}
nodeptr_t treeRotateLeft(nodeptr_t root) {
    nodeptr_t right = root-&gt;right;
    root-&gt;right = right-&gt;left;
    right-&gt;left = root;

    left-&gt;height = max(treeHeight(left-&gt;left), treeHeight(left-&gt;right))+1;
    right-&gt;height = max(treeHeight(right-&gt;left), treeHeight(right-&gt;right))+1;

    return right;
}
//这是标准左旋和标准右旋的代码 其实本身非常简单 后面的四种平衡操作都是对两种旋转的应用
//平衡实现
nodeptr_t treeRebalance(nodeptr_t root) {
    int factor = treeGetBalanceFactor(root);
    if(factor &gt; 1 &amp;&amp; treeGetBalanceFactor(root-&gt;left) &gt; 0) // LL
        return treeRotateRight(root);
    else if(factor &gt; 1 &amp;&amp; treeGetBalanceFactor(root-&gt;left) &lt;= 0) { //LR
        root-&gt;left = treeRotateLeft(root-&gt;left);
        return treeRotateRight(temp);
    } else if(factor &lt; -1 &amp;&amp; treeGetBalanceFactor(root-&gt;right) &lt;= 0) // RR
        return treeRotateLeft(root);
    else if((factor &lt; -1 &amp;&amp; treeGetBalanceFactor(root-&gt;right) &gt; 0) { // RL
        root-&gt;right = treeRotateRight(root-&gt;right);
        return treeRotateLeft(root);
    } else { // Nothing happened.
        return root;
    }
}
</code></pre>
<p>That's how we get AVL code, but we're trying to get the balance out of the insertion and deletion, which means we're going to fix the insertion and delete codes based on these functions, and it's easier to wrap them up.</p>
<pre><code class="language-c">void treeInsert(nodeptr_t *rootptr, int value)
{
    nodeptr_t newNode;
    nodeptr_t root = *rootptr;

    if(root == NULL) {
        newNode = malloc(sizeof(node_t));
        assert(newNode);

        newNode-&gt;data = value;
        newNode-&gt;left = newNode-&gt;right = NULL;

        *rootptr = newNode;
    } else if(root-&gt;data == value) {
        return;
    } else {
        if(root-&gt;data &lt; value)
            treeInsert(&amp;root-&gt;right,value);
        else
            treeInsert(&amp;root-&gt;left,value)
    }

    treeRebalance(root);//递归使用平衡树的代码 这个代码递归执行了很多次
}
//
void treeDelete(nodeptr_t *rootptr, int data)
{
    nodeptr_t *toFree; // 拜拜了您呐
    nodeptr_t root = *rootptr;

    if(root) {
        if(root-&gt;data == value) {
            if(root-&gt;right) {
                root-&gt;data = treeDeleteMin(&amp;(root-&gt;right));
            } else {
                toFree = root;
                *rootptr = toFree-&gt;left;
                free(toFree);
            }
        } else {
        if(root-&gt;data &lt; value)
            treeDelete(&amp;root-&gt;right,value);
        else
            treeDelete(&amp;root-&gt;left,value)
        }

        treeRebalance(root);
    }
}
</code></pre>
<h3>Multiple-road search for tree B</h3>
<p>The number of numbers in front of each node limits the storage of an element to two children at each node, which will certainly lead to an overload of memory at the time of the big file storage, and we need access to hard drives, which are much faster than the memory, and the memory access speed is much less than the built-in cache, because for this reason the multi-road search tree we're introducing is based on the number of elements that each node can store and the number of children he has, and we're looking at 2,3 trees, 2,3 and 4 trees B and B+ below. </p>
<h4>2-3 trees</h4>
<p>By definition, each node has two or three children, two and three, respectively, and two have one element and two children (or two children) and three have one or two elements and three children (or none). </p>
<p>Similar to the dident tree, the 2 node requires that the left tree be smaller than the root, and the right subtree is larger than the root, unlike the 2 node cannot have a child, the 3 node principle is similar, the left node contains smaller elements, the right subtree contains larger elements, the middle tree contains elements between them. </p>
<p>And we're asking all the leaves of the two-3 tree to be on the same plane, and it's clear that the two-3 tree makes it more difficult to insert and delete. </p>
<p><strong>Insert Category</strong></p>
<p>Like the tree that sorted the fork, the insertion can only occur at the leaf node.</p>
<p>For the empty tree, insert a two-point point. </p>
<p>Insert the node into a two-point leaf. We need to upgrade it to three and correct the left-and-right relationship.</p>
<p>Insert node into a three-point leaf.</p>
<h4>Delete Category</h4>
<p>Remove the leaves node from the three nodes.</p>
<p>The removal of the leaves of the two nodes led to the creation of the synthesis node, which led to the cessation of the two-three trees, which required more complex processing, continued classification, and no time was wasted here.</p>
<h4>2-3-4 trees</h4>
<p>By definition, the expansion of the concept of two to three trees is more complicated, so that you can understand the concept.</p>
<h4>B tree.</h4>
<p>B tree is a balanced multi-road search tree, and the two preceding are unique examples of B tree, and the largest number of children at the nodes is called B numbers, order B numbers are introduced to deal with swap memory and extras.</p>
<p>Our idea is to adjust the number of B trees to the size of the memory, and the larger step can have a lower height, just to access the heights in the root memory. </p>
<h4>B+ Tree</h4>
<p>The B+ tree is an improvement of the B tree, which is no longer part of the tree we studied before, and his advantage is to be able to find the area, and to correct the problem that B tree can only look at from the root. </p>
<h3>Fragmented List Search/Hashi Table Summary</h3>
<p>In the search methods that are in front of us, whether it's sequence or disorder, linear or tree, the comparison is an inevitable part of the search, but it's not really possible to avoid it. The answer is yes, storage = f (key) -- this is a viable new storage technology -- hash technology -- we're building a mapf between keywords and storage positions, which is called a hash function, and hash technology is storing records in a continuous space called a rash table. </p>
<p>Hash, this is a very hot word that appears in the classroom of the data structure.</p>
<p>In essence, hash technology is both a storage and a search method, and there's no logical relationship between his data elements, and he's a search-oriented structure. </p>
<p>It's obvious that hash technology is not suitable for a single keyword multirecording situation, and it's not suitable for a range search. </p>
<p>Ideally, each hash function should be addressed to a different address, but the ideal is the ideal of a collision with a multi-key word-mail address. </p>
<h3>The construction of a hash function</h3>
<p>What is a good hash function, and we have a few basic rules, which are simple to calculate to increase efficiency, and evenly distributed addresses to avoid too many conflicts, based on the following methods:</p>
<h4>Direct Location</h4>
<p>For example, if we want to count the number of people of different ages, we can use age as the address directly, if we count the number of births in different years, we can use the year as the address to consider a linear function of the keyword. </p>
<p>These functions are a little simple, even, non-conflict, but need to be first to know the distribution of keywords, so they're not very common.</p>
<h4>Digital Analysis Method</h4>
<p>The key is taken from the top four, which are often used for ID checks, ID numbers, bank cards, of course, and also for the description of the key word, and for a balanced distribution.</p>
<h4>Square-based</h4>
<p>The original keyword squared and then a few places in the middle, like three, which is appropriate for a situation where the number of digits is not very large, but rather a few.</p>
<h4>Collapse</h4>
<p>Split the keywords into the same bits and then make a request and as an address. </p>
<p>One fold may be uneven, perhaps one that can be folded from the other side, and two combined will be more even.</p>
<p>It's a big bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a sort of a little bit of a sort of a sort of a sort of a sort of a sort of a sort of a sort of a sort of a little bit of a sort of a sort of a sort of a sort of a sort of a sort of a sort of a sort of a sort of a sort of a sort of a thing.</p>
<h4>Save the remaining number</h4>
<p>It's the mod function, and mathematicians tell us when the table is long, you usually choose to separate the smallest mass close to m or not to contain a combination of 20 or less progeny.</p>
<h4>Random number</h4>
<p>The random numbers are usually pseudo-random, and there are algorithms that generate random numbers. </p>
<p>The principle of random number generation does not have to be said here, but it's also used in square numbers.</p>
<h3>Dealing with the hash conflict</h3>
<p>When we've found the conflict, how do we fix it? There's a way.</p>
<h4>Open Locations</h4>
<p>When you're in conflict, choose the next empty hash address, and if the watch is big enough, you're not afraid to find it.</p>
<p>hi(key) = (h(key)+di) mod m</p>
<p>This is the core open location formula.     </p>
<p>Linear detection: di = i</p>
<p>Square detection: di = ±i2 (+12, -12, +22, -22) Square detection is to avoid accumulation effects and better occupy the whole bulk list</p>
<p>Random detection: di is random number by time or by anything.</p>
<h4>Re-hash Function Method</h4>
<p>* h2(key) makes several hash functions always changing functions always solves problems</p>
<h4>Chain Address Method</h4>
<p>I'm throwing a single watch here, one element plus one, and I'm gonna find this address and I'm gonna go over it again.</p>
<h4>Public Spill Areas Act</h4>
<p>All address conflicts, separate spill areas, and if Hathy doesn't come to my spill area, it's good to have a few conflict data.</p>
<h3>Fragmented List Searches</h3>
<p>With the ideas that are ahead, this should not be the problem. </p>
<pre><code class="language-c">#define m 16	//  哈希表/散列表长度
typedef int KeyType;
typedef int InfoType;
//散列表定义
typedef struct
{
	KeyType key;
	InfoType otherinfo;
}HashTable[m];

//散列表的查找
int SearchHash(HashTable HT, KeyType key)
{
	int HO = key % 13;  //根据散列函数计算散列地址
	if (HT[HO].key == 0)  return -1;		// 若单元为空， 则所查元素不存在
	else if (HT[HO].key == key) return HO;
	else
	{
		//按照线性探测法计算下一个散列地址Hi
		for (int i = 1; i &lt; m; i++)
		{
			int Hi = (HO + i) % m;
			if (HT[Hi].key == 0)  return -1;		// 若单元为空， 则所查元素不存在
			else if (HT[Hi].key == key) return Hi;
		}
		return -1;
	}
}

//散列表的插入
int InsertHash(HashTable HT, KeyType key)
{
	int HO = key % 13;		//根据散列函数计算散列地址
	if (HT[HO].key == 0)	// 若单元为空， 则所查元素不存在
	{
		HT[HO].key = key;
		return 0;
	}
	else
	{
		//按照线性探测法计算下一个散列地址Hi
		for (int i = 1; i &lt; m; i++)
		{
			int Hi = (HO + i) % m;
			if (HT[Hi].key == 0) 	// 若单元为空， 则所查元素不存在
			{
				HT[Hi].key = key;
				return 0;
			}
		}
		return -1;					//散列表已满
	}
}


int main()
{
	//初始化
	HashTable HT;
	for (int i = 0; i &lt; m; i++)
	{
		HT[i].key = 0;
	}

	//插入
	InsertHash(HT, 19);
	InsertHash(HT, 14);
	InsertHash(HT, 23);
	InsertHash(HT, 1);
	InsertHash(HT, 68);
	InsertHash(HT, 20);
	InsertHash(HT, 84);
	InsertHash(HT, 27);
	InsertHash(HT, 55);
	InsertHash(HT, 11);
	InsertHash(HT, 10);
	InsertHash(HT, 79);

	//遍历散列表
	printf(&quot;按散列地址排列：&quot;);
	for (int i = 1; i &lt;m; i++)
	{
		printf(&quot;%d,&quot;, HT[i].key);
	}

	//查找
	int n;
	printf(&quot;\n请输入要查找的数：&quot;);
	scanf(&quot;%d&quot;, &amp;n);
	int result = SearchHash(HT, n);
	printf(&quot;\n要查找的数在散列表中的地址为：%d  \n&quot;, result);
}
</code></pre>
<p>The array symbol is actually just a representation. </p>
<p>All Hash before us is for numbers, because computers are binary, so if the code can be converted to numbers, Hash can be for all keywords.</p>
<h2>Sort</h2>
<p>We've mentioned a concept a million times before, and we've been looking for information on the Internet, and we've been talking about orderly concepts, so we've been asking for very important algorithms, sorting questions.</p>
<h3>Basic concepts and classifications of ranking</h3>
<p>The core of the sorting is that we make the sequence meet non-ingressive or non-ingressive relationships by key code, and we usually use non-ingressive sequences.  </p>
<p>The sorting of multiple keywords is essentially a supersizing of the sort of keywords, sometimes directly simplified by the word-linking, so we focus on the sorting of the word.</p>
<p>When two keywords appear to be sorted in equal terms, they are drawn out.<strong>Sort stability</strong>The concept, which means that if two people are doing the same thing, the elements that were in front of the ranking should be in front of the ranking, and then stability, and the stability of the ranking algorithm, is a concept that we need to consider later.</p>
<p>We've been able to get a brief picture of the differences between data memory and memory in the previous study, and this is a problem that happens when you sort out.<strong>Sort inside and outside</strong>The concept of a single, and then it's just about sorting out the inside.</p>
<p>The performance of the internal ranking algorithm is measured mainly from three angles <strong>Time performance, auxiliary space, algorithm complexity.</strong> The complexity here is the complexity of algorithms, not the complexity of time.</p>
<p>By the main operation of sorting, we sort sort of insert sort of sort of swap sort of select sort of sort of group group of four big Category</p>
<p>Based on the complexity of algorithms, we divide them into simple algorithms.<em>A sort of bubble-bearing, a simple selection of sorting, a direct insertion of sorting</em>     Improved algorithm <em>It's got Hill sorted, stacked, sorted, sorted quickly.</em></p>
<p>The basics of sorting are a linear table, because sorting often requires the exchange of elements, so we use some encapsulation elements, and then we don't explain.</p>
<h3>Bubble Sort</h3>
<p>And that's the simplest sort of sort of sort of thing, knowing him at the language first stage, his core idea is to keep two more than one key words in the next, and if it's reversed, then it's exchanged until there's no reverse sequence, and the standard bubble sorting code is as follows, and time is complicated. degrees&#36;n^{2}&#36;  It's stable.</p>
<pre><code class="language-c">void bubble_sort(int a[], int n)
{
    int i,j,temp;
    for (j=0;j&lt;n-1;j++)
    {
        for (i=0;i&lt;n-1-j;i++)
        {
            if(a[i]&gt;a[i+1])
            {
                temp=a[i];
                a[i]=a[i+1];
                a[i+1]=temp;
            }
        }
    }
}
//有时候冒泡排序会做一些无意义的比较 我们可以选择增加flag来避免有序情况下的判断（如果有一轮已经发现没有发生任何交换 终止算法）
</code></pre>
<h3>Simple Select Sort</h3>
<p>And that's the sort of selection we're talking about most often, finding the smallest one in the front and then revolving, and the complexity of time.&#36;n^2&#36; But it's actually a little better. It's stable.</p>
<pre><code class="language-c">void select_sort(int R[],int n)
{
    int i,j,k,index;
    for(i=0;i&lt;n-1;i++)
    {
        k=i;
        for(j=i+1;j&lt;n;j++)
        {
            if(R[j]&lt;R[k])
                k=j;
        }
        index=R[i];
        R[i]=R[k];
        R[k]=index;
    }
}
</code></pre>
<h3>Insert Sorting Directly</h3>
<p>The idea of inserting sorting is to insert a record into a chart that is already in order, and get a new order sheet, and the first two elements will be put in the first sorting, and then the right position will be selected, and the insertion process will be complicated. degrees&#36;n^{2}&#36;  Steady</p>
<pre><code class="language-c">void insertion_sort(int number[],int n)
{
    int i=0,j=0,temp=0;
    for(i=1;i&lt;n;i++)
    {
        temp=number[i];
        j=i-1;
        while(j&gt;=0&amp;&amp;temp&lt;number[j])
        {
            number[j+1]=number[j];
            j--;
        }
        number[j+1]=temp;
    }
}
</code></pre>
<h3>Shell Sort</h3>
<p>In the course of the evolution of the ranking algorithm, the three algorithms and their optimization are being mainstreamed over a long period of time, and because of the time complexity, it was thought that the time complexity of the ranking algorithm could not be lower than that of the time.&#36;n^{2}&#36; Fortunately, the complexity of the time was finally broken by some scientists.</p>
<p>Hill sorted out as an optimisation for direct insertion.&#36;n^{2}&#36;The increase, if it reduces the n, it will effectively reduce the time complexity, so he chooses to optimize the original sequence into a small sub-series, and when the sequence is essentially organized, then the core point is that the sequence is inserted directly into the sequence. <strong>Basically orderly.</strong> What is it? Let's see the code. </p>
<p>The optimisation subseries is taken by the k-point, and divided into groups, and thus loses stability.</p>
<pre><code class="language-c">void ShellSort(int L[].int n){
    int i,j;
    int increment = n;
    int temp;
    do{
        increment = increment/3+1;
        for(i=increment+1;i&lt;=n;i++){
            if(L[i]&lt;L[i-increment]){
                temp=L[i];
                for(j=i-increment;j&gt;0&amp;&amp;temp&lt;L[j];j-=increment){
                    L[j+increment]=L[j];
                }
                L[j+increment]=temp;
            }
        }
    }
    while(increment&gt;1);
}
//这段代码L[0]位置是空的 不存储数据 很明显 increment的选取非常重要 我们这一的知识一种方法 仅供参考
</code></pre>
<p>The core of Hill's sorting algorithm is the direct insertion of spacing comparisons to leaps, and we compare the elements that are increment and directly sequenced, and then narrow it again after the round of dowhile cycles, and actually, the smaller the sorting that needs to be done after the previous rounds, which is the core of Hill's sorting, and this is the way to optimize it, and we're able to reduce the time complexity to n^1.5, although progress is not significant, the speed of breaking through the slow sequencing is important, and the average complexity changes, but it doesn't break the nlogn.</p>
<h3>Stack Sort Heap Sort</h3>
<p>The stacking is an improvement in the simple selection sorting, and in the simple selection sorting, we actually do a lot of replicating comparisons after the first comparison was completed, which leads to excessive time complexity, and the idea of stacking is to use the new data structure of the stack to complement, instability, average complexity. degrees&#36;nlogn&#36;The limit is the average.</p>
<p><strong>The pile is a full-blown fork tree with the following characteristics:</strong>  </p>
<p>And under the condition of piles, we're talking about a nature in the complete fork tree.</p>
<p>For a full binary tree with n nodes, number the node in a stratification order for any node i</p>
<p>At 1 p.m., he was root nodes without parents.&gt;Parents at 1st</p>
<p>If 2i&gt;Node i has no left child.</p>
<p>If 2i+1&gt;No, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no. No, no, no, no, no. No, no, no. No. I has no right, no, no, no, no, no, no, no, no, no, no.</p>
<p>Obviously, if the large and small tops are to be stratified, it's a roughly sequenced array, and the sort of sorting is to make the original sequence a large pile, move the top elements to the end, and then the rest of the elements become a pile of tops, and repeat them, and we can get an orderly sequence, and the questions that we're wondering about can be understood by code.</p>
<pre><code class="language-c">void swap(int* a, int* b) {
    int temp = *b;
    *b = *a;
    *a = temp;
}//懂得都懂 后面那么多结点交换 这样轻松一些
void max_heapify(int arr[], int start, int end) {
    //建立父节点指标和子节点指标
    int dad = start;
    int son = dad * 2 + 1;
    while (son &lt;= end) { //若子节点指标在范围内才做比较
        if (son + 1 &lt;= end &amp;&amp; arr[son] &lt; arr[son + 1]) //先比较两个子节点大小，选择最大的
            son++;
        if (arr[dad] &gt; arr[son]) //如果父节点大于子节点代表调整完毕，直接跳出函数
            return;
        else { //否则交换父子内容再继续子节点和孙节点比较
            swap(&amp;arr[dad], &amp;arr[son]);
            dad = son;
            son = dad * 2 + 1;
        }
    }
}
void heap_sort(int arr[], int len) {
    int i;
    //初始化，i从最后一个父节点开始调整
    for (i = len / 2 - 1; i &gt;= 0; i--)
        max_heapify(arr, i, len - 1);
    //先将第一个元素和已排好元素前一位做交换，再从新调整，直到排序完毕
    for (i = len - 1; i &gt; 0; i--) {
        swap(&amp;arr[0], &amp;arr[i]);
        max_heapify(arr, 0, i - 1);
    }
}
//修正完以后的二叉树进行层序遍历就可以有序了
</code></pre>
<h3>Sorting Merging Sort</h3>
<p>And so, if you have an entire series of sequenced tables, and if you have an initial record of one, you have an sequence of one, and then two of them have a sequence of one or two lengths, and then you repeat it, and you get an sequence of sequences of two, which is a sequence of two, which is a sequence of two, which is a sequence of two, which is a stable, complex average. degrees&#36;nlogn&#36; The limit is average.</p>
<p>When you merge, you actually track two decimal groups with two fingers and then you create separate spaces to store the results of the amalgamation.</p>
<pre><code class="language-c">void merge_sort_recursive(int arr[], int reg[], int start, int end) {
    if (start &gt;= end)
        return;
    int len = end - start, mid = (len &gt;&gt; 1) + start;
    int start1 = start, end1 = mid;
    int start2 = mid + 1, end2 = end;
    merge_sort_recursive(arr, reg, start1, end1);
    merge_sort_recursive(arr, reg, start2, end2);
    int k = start;
    while (start1 &lt;= end1 &amp;&amp; start2 &lt;= end2)
        reg[k++] = arr[start1] &lt; arr[start2] ? arr[start1++] : arr[start2++];
    while (start1 &lt;= end1)
        reg[k++] = arr[start1++];
    while (start2 &lt;= end2)
        reg[k++] = arr[start2++];  //这三个while循环就是对拆分双指针递归的实现
								//理解一下思路 我们从两个数组里从两边的头开始比大小 找小的塞进reg里面 然后下一位
    for (k = start; k &lt;= end; k++)
        arr[k] = reg[k]; //把reg临时存放的数据扔回去方便递归回去调用
}
void merge_sort(int arr[], const int len) {
    int reg[len];
    merge_sort_recursive(arr, reg, 0, len - 1);
}
//这里只是进行了一次调用 方便我们进行前面函数的递归操作
</code></pre>
<h3>Counting Sort</h3>
<p>The ranking of numbers is not based on a comparative sorting algorithm.</p>
<p>The core is to convert the data values entered into key to be stored in extra-created array space. As a sort of linear time complexity, the order of count requires that the data entered be integers with a defined range. Its basic idea is that each element of the given input series x determines the number of median values of the series less than the equivalent of the x element, and then it is stored directly at the correct location of the final sorting sequence.</p>
<h3>Bucket Sort</h3>
<p>The sorting of drums (Bucket sort) or so-called box sorting is a sorting algorithm that works on the basis of the distribution of arrays into a limited number of barrels. Each barrel is then sorted individually (possibly using alternative sorting algorithms or continuing to sort in a descending pattern), and the records in each barrel are then listed in order to remember the sequence.</p>
<h3>Base Sort Radix Sort</h3>
<p>Radix sort is a non-comparable integer sorting algorithm.</p>
<p>The rationale is to cut the whole number into a different number, and then compare it separately. The base figure can be sorted in LSD (Lest significant digital) or MSD (Most significant digital), where the LSD is sorted from the right side of the key value, whereas the MSD, in contrast, starts from the left side of the key value.</p>
<ul>
<li><strong>MSD</strong>: Sorting first from the top, and in each key word, by counting</li>
<li><strong>LSD</strong>: Sorting from lower to lower, with barrel sorting for each key word</li>
</ul>
<h3>Quick Sort</h3>
<p>The quick-sorting is the most basic sorting we've ever mentioned, the bubble-sorting upgrade, which is also made by constant comparison and movement, but he increases the distance between comparison and movement, and thus reduces the number of comparisons and exchanges.</p>
<p>Basic thought: to divide the pending records into two separate parts by a sort of sequence, one of which is smaller than the other, and thus sorting out two separate parts, and thus keeping the whole series in order, and he looks like he's sorting to Hill, but actually, it's not the same as the original, much bigger part, and we look at the code.</p>
<pre><code class="language-c">void QuickSort(Sqlist *L){
    Qsort(L,1,L-&gt;length);
}
//和归并一样 因为涉及到递归调用的问题我们添加了一个封装层
void Qsort(Sqlist *L,int low,int high){
    int pivot;
    if(low&lt;high){
        pivot=Partition(L,low,high); //用调用了一个函数 他的作用是选择一个关键词 是谁无所谓
							//然后找到一个位置让他左边都比他小 右边都比他大
        Qsort(L,low,pivot-1);
        Qsort(L,pivot+1,high);//两次递归调用
    }
}
int Partition(Sqlist *L,int low,int high){
    int pivotkey;
    pivotkey = L-&gt;r[low];
    while(low&lt;high){
        while(low&lt;high&amp;&amp;L-&gt;r[high]&gt;=pivotkey){
            high--;
        }
        swap(L,low,high); //就当这是一个封装好的函数就行 虽然C里没有 这里我们的核心是理解算法
        while(low&lt;high&amp;&amp;L-&gt;r[low]&lt;=pivotkey){
            low++;
        }
        swap(L,low,high); //就是从两端找元素 找到了就和选定的pivot交换 最后形成一个左小右大 high=low的时候OK了
    }
    return lowl;
}
//1 开始存元素的地方 别用前面那数组了 还是链表好用
</code></pre>
<p>The speed of sorting is unstable. &#36;nlogn&#36;  The worst case scenario (the original sequence) is the complexity of the time.&#36;n^{2}&#36;</p>
<p><strong>Optimizing pivot</strong></p>
<p>The more the value is close to the median of the whole, the less the algorithm will be calculated after it is, so we'll introduce a three-digit medium, nine-digit medium, and hopefully it'll be closer to the middle key.</p>
<p><strong>Optimizing exchange</strong></p>
<p>The swap function is replaced by a swap function that is not sealed, and it is hoped that there will be some operation savings here, but leaving the envelope will be closer to the bottom. </p>
<p><strong>Optimizing decimal schemes</strong></p>
<p>If you're not quick in sorting in decimal groups, you can just choose to insert a direct sorting in the order of the number of decimals.</p>
<p><strong>Optimizing Recursive</strong></p>
<p>The reverse is not a small consumption of computer performance, so there is an optimisation of tail return to improve performance.</p>
<h3>Final remarks</h3>
<p>And what's interesting is that this sort of algorithm is called fast-sequencing, and it's actually a big problem, and if there's a better sorting algorithm, he's not really sure, and it's actually the fastest way to get it, and it's the most efficient algorithm in the world, and it's hard to get it right.</p>
