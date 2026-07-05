---
title: "查找与排序算法：二分查找、散列表与排序实现"
title_en: "Search and Sorting Algorithms: Binary Search, Hash Tables, and Sorting Implementations"
date: 2024-12-18 21:37:43 +0800
categories: ["Programming", "Computer Science Fundamentals"]
tags: ["Learning Notes", "Algorithms", "Search", "Sorting"]
author: Hyacehila
excerpt: "整理顺序查找、二分查找、二叉排序树、散列表、排序算法和相关 C 语言实现。"
excerpt_en: "Covers sequential search, binary search, binary search trees, hash tables, sorting algorithms, and related C implementations."
mathjax: true
hidden: true
permalink: '/blog/2024/12/18/search-and-sorting-algorithms/'
---
## 查找

这是一类非常巨大的问题 Search是任何一个程序员都要会并且经常使用的工具 也是现代科学发展的核心 有的可能专门就是学习查找工具的

### 查找概论

所有需要被查询的数据所在的集合称为**查找表** Search Table

**关键词** key是 数据元素中某一项  他可以标识一个数据元素 也可以标识一个字段（关键码）

如果关键词可以唯一标识一个记录 则是称他是 **主关键词** Primary Key 反之称为次关键词 Secondary Key 他们也对应关键码

所谓的查找就是根据一个值 找到对应的记录

我们在查找成功的时候返回位置 否则返回空指针（一般）

按照操作方式 我们把查找表分为静态查找表和动态查找表

**静态查找表** 就是我们一般意义上的查找 在一堆数据里面找自己想要的

**动态查找表** 会在查找的过程中插入或者删除数据元素  

为了提高查找的效率 我们往往为查找操作设置合适的数据结构  合适的数据结构的选取是为了在查找的时候获得更高的查找性能

合适的结构 就是我们在后面要讨论的 一般情况下 静态查找选择线性表 动态查找考虑二叉排序树 部分特殊情况选择散列表

### 顺序表查找

#### 顺序查找

此时数据元素是无序的线性表 我们没什么好用的技巧 挨个比对是唯一的办法 代码示意如下

```c
int linear_search(int arr[N], int value) {
    int i;
    for (i = 0; i < N; i++) {
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
```

#### 有序表查找 

如果原本的元素按照某个顺序排列 这对于我们的查找就有很大的帮助了

#### 二分查找

```c
int search(int nums[], int size, int target) //nums是数组，size是数组的大小，target是需要查找的值
{
    int left = 0;
    int right = size - 1;	// 定义了target在左闭右闭的区间内，[left, right]
    while (left <= right) {	//当left == right时，区间[left, right]仍然有效
        int middle = left + ((right - left) / 2);//等同于 (left + right) / 2 /符号自动取整
        if (nums[middle] > target) {
            right = middle - 1;	//target在左区间，所以[left, middle - 1]
        } else if (nums[middle] < target) {
            left = middle + 1;	//target在右区间，所以[middle + 1, right]
        } else {	//既不在左边，也不在右边，那就是找到答案了
            return middle;
        }
    }
    return -1;
}
//原理非常容易 处理好边界就可以了 不复杂
```

#### 插值查找

能从一半的地方开始 别处不行吗？ 数学家们给出了自己的答案 插值查找公式

```c
mid = low + (high-low)*(key-a[low])/(a[high]-a[low]);
```

使用插值公式 有时候能增加查找的效率 比如数组很长并且均匀的时候

#### 斐波那契查找

也是从分隔点入手的 由于F（n）= F（n-1）+ F（n-2） 所以我们选择把数组分成这样的两部分

```c
int Fibonacci_Search(int *a, int key, int n)
{
	int i, low = 0, high = n - 1;
	int mid = 0;
	int k = 0;
	int F[ARRSIZE];
	InitFibonacci(F);
	while (n > F[k] - 1)          //计算出n在斐波那契中的数列  
	{
		++k;
	}
	for (i = n; i < F[k] - 1; ++i) //把数组补全  最大补全到后面的位置
	{
		a[i] = a[high];
	}
	while (low <= high)
	{
		mid = low + F[k - 1] - 1;  //根据斐波那契数列进行黄金分割  
		if (a[mid] > key)
		{
			high = mid - 1;
			k = k - 1;
		}
		else if (a[mid] < key)
		{
			low = mid + 1;
			k = k - 2;
		}
		else
		{
			if (mid <= high) //如果为真则找到相应的位置  
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
```

### 线性索引查找

如果数据量非常庞大 让他们有序排列是肯定不可能的 但是我们能否让无序的排列有一点规律可言吗 当然可以 图书馆的书都是使用索引这个中介方便我们查找的 把关键词和一个对应的记录绑定 我们无序排列也能有序查找 

索引结构一般分为 **线性索引 树形索引 多级索引** 我们这里着重介绍线性索引结构

#### 稠密索引

每一个数据集中的记录都对应一个索引项 索引项一定是按照关键码有序排列的 稠密索引可能导致的问题是大量数据导致索引项过多

#### 分块索引

图书馆的图书本质上就是分块索引 我们把原本的图书分块 块内无序（架子上的书） 块间有序  我们在块内查找具体元素的时候选择最基本的顺序查找 分块索引在数据库中非常的常用 因为他有专人在维护并且人力消耗不大

#### 倒排索引

这其实是搜索引擎比较常用的技术 很容易发现 搜索引擎的效率好像高的可怕 我们这里只会进行简单的介绍 真正的搜索引擎技术需要长期的学习和积累

我们把可能出现的所有关键词单独提取 并且找到他们对应的原始文件编号 此时再输入关键词的时候 搜索效率就会非常高 因为我们可以把提取的关键词进行排序或者制造索引结构

这种索引技术就被称为倒排索引 inverted index 他的核心结构是关键码和记录号表 

### 二叉排序树

我们现在来尝试一下 **动态查找表**    我们希望一方面查找效率不错 另一方面能够方便插入和删除 正如本章的标题 我们希望借助二叉树来实现 在构造二叉树方面 我们把新元素和原本树里面的元素一层一层的比较大小 小的放左边 大的选右边 一层一层来 最后得到的二叉树进行中序遍历的时候就是一个有序的数列 这就是二叉排序树 Binary Sort Tree 他有一下性质

* 如果左子树不空 左子树所有结点的值小于他跟结构的值
* 如果右子树不空 右子树所有结点的值大于根节点的值
* 左右子树分别为二叉排序树

很明面 递归是我们研究二叉树不能错过的话题 构造二叉排序树不是为了排序 而是为了方便查找和插入与删除 下面是一些代码实现

```c
typedef int DataType;
typedef struct BST_Node {
    DataType data;
    struct BST_Node *lchild, *rchild;
}BST_T, *BST_P;
//我们先默认有一棵二叉排序树 建立放到插入后面讲 很快就会理解意思
BST_P SearchMin(BST_P root)
{
    if (root == NULL)
        return NULL;
    if (root->lchild == NULL)
        return root;
    else  //一直往左孩子找，直到没有左孩子的结点  
        return SearchMin(root->lchild);
}
//查找最大原理也非常的简单 略去了
BST_P Search_BST(BST_P root, DataType key)
{
    if (root == NULL)
        return NULL;
    if (key > root->data) //查找右子树  
        return Search_BST(root->rchild, key);
    else if (key < root->data) //查找左子树  
        return Search_BST(root->lchild, key);
    else
        return root;
}
//递归是查找算法的核心 学习这些也会帮助我们理解递归
void Insert_BST(BST_P *root, DataType data)
{
    //初始化插入节点
    BST_P p = (BST_P)malloc(sizeof(struct BST_Node));
    if (!p) return;
    p->data = data;
    p->lchild = p->rchild = NULL;

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
        if(data < troot->data){
            troot = troot->lchild;
        }
        else{
            troot = troot->rchild；
        }
    }
    if (data < tnode->data)
        tnode->lchild = p;
    else
        tnode->rchild = p;
}
//所谓的插入就是查找一个合适的地方添加进去
void CreateBST(BST_P *T, int a[], int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        Insert_BST(T, a[i]);
    }
}
//所谓的建立就是重复插入的过程 非常简单
void DeleteBSTNode(BST_P *root, DataType data)
{
    BST_P p = *root, parent = NULL, s = NULL;

    if (!p) return;

    if (p->data == data) //找到要删除的节点了
    {
        /* It's a leaf node */
        if (!p->rchild && !p->lchild) 
            *root = NULL;

        // 只有一个左节点
        else if (!p->rchild&&p->lchild) 
            *root = p->lchild;

        // 只有一个右节点
        else if (!p->lchild&&p->rchild) 
            *root = p->rchild;

        //左右节点都不空 里面是一个复杂的判断过程 
        else 
        {
            s = p->rchild;
            /* the s without left child */
            if (!s->lchild)
                s->lchild = p->lchild;  //完成接树的过程
            
            /* the s have left child */
            else 
            {
                /* find the smallest node in the left subtree of s */
                while (s->lchild) 
                {
                    /* record the parent node of s */
                    parent = s;
                    s = s->lchild;
                }
                parent->lchild = s->rchild;
                s->lchild = p->lchild;
                s->rchild = p->rchild;
            }
            *root = s;
        }
        free(p);
    }
    else if (data > p->data) //向右找
        DeleteBSTNode(&(p->rchild), data);
    else if (data < p->data) //向左找
        DeleteBSTNode(&(p->lchild), data);
}
//删除其实比较复杂 你得分清是不是叶子结点 如果要删除的结点有子孙 我们需要怎么变化我们的二叉树 
//删除代码包括了我们进行查找的过程 后面的递归就是如此
```

二叉排序树的主要部分就是这些 我们很容易发现一个问题 我们希望树比较平衡 深度和完全二叉树一致 这样能减少查找消耗的判断次数 也就是平衡二叉树问题

### 平衡二叉树 AVL树

平衡的意思很简单 就是希望所有结点左子树和右子树的高度最多相差1 我们把二叉树结点左子树深度减去右子树深度的值称为平衡因子 BF 对于AVL树平衡因子只能为 1 0 -1  当然 如果不是排序二叉树 平衡二叉树的前提条件就没了 

我们称距离插入结点最近的 且平衡因子绝对值大于1的结点为根的子树被我们称为最小不平衡子树 

平衡二叉树的构建的核心就是在构建二叉排序树的时候每当插入一个节点就检查一下平衡性是否还存在 如果平衡性被破坏 就在保持二叉排序树前提特性的前提下 调整最小不平衡子树各结点之间的逻辑关系 让他称为新的平衡子树 下面我们来介绍一下思路 用一个简单的例子来讲解

3 2 1 4 5 6 7 10 9 8 直接构成的二叉排序树如图所示  理论上优化后构成的如另一图所示 如何实现这个转换呢

我们从头开始考虑 什么时候出现了不平衡 这时候应该怎么办 

1 插入这个二叉树后 我们发现整棵树成为了最小不平衡子树 为了让他平衡 需要整体顺时针旋转 让2成为根节点 

4正常插入 5插入的时候 结点3出现最小不平衡子树 需要逆时针旋转 

增加结点6 整棵树再次成为了最小不平衡子树 以2为结点逆时针旋转

同理 7加入 10加入  9加入的时候再次不平衡 但是此时直接旋转解决不了问题了 这是因为最小不平衡子树的BF和他的子树的BF符号相反 所以需要进行两次旋转来解决问题

也就是四种情况   LL RR  LR RL 不难理解

这个旋转的思路就是AVL树的实现思路 在插入的过程中就着手解决 下面是代码示例

```c
struct node {
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
        return max(treeHeight(root->left),treeHeight(root->right)) + 1;
    }
}
//用来获得结点高度的函数 在后面进行什么删除或者插入的操作的时候记得更新高度这个量
int treeGetBalanceFactor(nodeptr_t root) {
    if(root == NULL)
        return 0;
    else
        return x->left->height - x->right->height;
}
//检测BF因子 当绝对值大于一的时候就应该进行一次修正
nodeptr_t treeRotateRight(nodeptr_t root) {
    nodeptr_t left = root->left; //保存新的根节点 也就是原本结点的左孩子
    root->left = left->right; // 将将要被抛弃的节点连接为旋转后的 root 的左孩子
    left->right = root; // 调换父子关系

    left->height = max(treeHeight(left->left), treeHeight(left->right))+1;
    right->height = max(treeHeight(right->left), treeHeight(right->right))+1;
    
    return left; //返回的是这一部分的新的根节点
}
nodeptr_t treeRotateLeft(nodeptr_t root) {
    nodeptr_t right = root->right;
    root->right = right->left;
    right->left = root;

    left->height = max(treeHeight(left->left), treeHeight(left->right))+1;
    right->height = max(treeHeight(right->left), treeHeight(right->right))+1;

    return right;
}
//这是标准左旋和标准右旋的代码 其实本身非常简单 后面的四种平衡操作都是对两种旋转的应用
//平衡实现
nodeptr_t treeRebalance(nodeptr_t root) {
    int factor = treeGetBalanceFactor(root);
    if(factor > 1 && treeGetBalanceFactor(root->left) > 0) // LL
        return treeRotateRight(root);
    else if(factor > 1 && treeGetBalanceFactor(root->left) <= 0) { //LR
        root->left = treeRotateLeft(root->left);
        return treeRotateRight(temp);
    } else if(factor < -1 && treeGetBalanceFactor(root->right) <= 0) // RR
        return treeRotateLeft(root);
    else if((factor < -1 && treeGetBalanceFactor(root->right) > 0) { // RL
        root->right = treeRotateRight(root->right);
        return treeRotateLeft(root);
    } else { // Nothing happened.
        return root;
    }
}
```

实现AVL的代码就是这样的  但是我们是在插入和删除的过程中导致失去平衡 也就是说我们要根据这些函数修正插入和删除代码 封装后轻松了很多

```c
void treeInsert(nodeptr_t *rootptr, int value)
{
    nodeptr_t newNode;
    nodeptr_t root = *rootptr;

    if(root == NULL) {
        newNode = malloc(sizeof(node_t));
        assert(newNode);

        newNode->data = value;
        newNode->left = newNode->right = NULL;

        *rootptr = newNode;
    } else if(root->data == value) {
        return;
    } else {
        if(root->data < value)
            treeInsert(&root->right,value);
        else
            treeInsert(&root->left,value)
    }

    treeRebalance(root);//递归使用平衡树的代码 这个代码递归执行了很多次
}
//
void treeDelete(nodeptr_t *rootptr, int data)
{
    nodeptr_t *toFree; // 拜拜了您呐
    nodeptr_t root = *rootptr;

    if(root) {
        if(root->data == value) {
            if(root->right) {
                root->data = treeDeleteMin(&(root->right));
            } else {
                toFree = root;
                *rootptr = toFree->left;
                free(toFree);
            }
        } else {
        if(root->data < value)
            treeDelete(&root->right,value);
        else
            treeDelete(&root->left,value)
        }

        treeRebalance(root);
    }
}
```

### 多路查找树 B树

前面的数限制在每个结点存储一个元素 二叉树限制每个结点最多有两个孩子 这在进行大文件存储的时候肯定会导致内存溢出 我们需要访问硬盘 而硬盘的访问速度是远低于内存的 内存的访问速度是远低于内置缓存的 因为这个原因我们引入的多路查找树  根据每个结点能存储的元素数目和他的孩子数目 我们在下面研究2 3树 2 3 4树 B树 和B+树 

#### 2-3树

顾名思义  每个结点具有两个到三个孩子 他们分别是2结点和3结点 2结点包含一个元素和俩孩子（或者0个孩子） 3结点包含一大一小两个元素和三孩子（或者0个孩子） 

与二叉排序树类似的是 2结点要求左子树小于根 右子树大于根 不同的是2结点不能有一个孩子  3结点原理类似 左子树包含较小元素 右子树包含较大元素 中间树包含介于两者之间的元素 

并且我们要求2-3树的所有叶子结点在同一个平面上 很明显 2-3树增大了我们进行插入和删除的难度 

**插入情况分类**

正如二叉排序树一样 插入只能发生在叶子结点

对于空树 插入一个二结点就可以 

插入结点到一个二结点的叶子上 我们需要把它升级为3结点  并且修正左右关系

插入结点到一个三结点的叶子上 需要拆分结点 移动层数……

#### 删除情况分类

删除三节点的叶子结点 这很简单 删除以后更改他为二结点

删除二结点的叶子结点 导致单元素节点的产生 这就是导致不再是2-3树 需要比较复杂的处理 继续分类研究 在这里就不浪费时间了

#### 2-3-4树

顾名思义 是2-3树概念的拓展 更加复杂 了解概念就可以

#### B树

B树是一种平衡的多路查找树 前面的两种都是B树的特例  结点的最大的孩子数目称为B数的阶 order B数被引入就是为了处理交换内存和外存

我们的思路就是根据内存的大小调整B树的阶数 更大的阶数可以有更低的高度 只要根结点在内存中 访问高度次外存就可以了 

#### B+树

B+树是一种B树的改良 他已经不属于我们前面研究的树的范畴 他的好处是便于带范围的查找 修正了B树只能从根结点查找的问题 

### 散列表查找/哈希表概述

在前面的查找方式里面 无论是顺序还是无序 线性还是树形 比较都是查找不可避免的一部分 但是比较真的不能被避免吗 能否直接用关键词就得到存储位置吗 答案是肯定的 存储位置=f（key） 这是一种可行的新的存储技术 散列技术 我们在关键字和存储位置之间建立映射f 这种映射就被称为散列函数 也称为Hash函数 散列技术把记录存储在一块连续空间里面 这块空间被称为Hash table 

Hash  这个目前非常热的词汇 在数据结构的课堂上出现了

本质上 散列技术既是一种存储方法 也是一种查找方法 他的数据元素之间没有任何的逻辑关系 他就是一种面向查找的结构 

很明显的 散列技术不适用于单关键词多记录的情况 不适合范围查找 

理想状况下 散列函数每一个关键词都应该对应不同的地址 但是理想只是理想 多关键词单地址的collision情况不可避免 此时两个key被称为synonym 如何精妙的控制散列函数避免collision的发生是非常重要的 

### 散列函数的构造

什么是一个好的散列函数 我们有几条基本规则 计算简单来增加效率 地址分布均匀来避免过多的冲突发生 基于这些规则有以下的方法

#### 直接定址法

举个例子 如果我们要统计不同年龄人口数目 可以直接用年龄作为地址  如果统计不同年份的出生人数 可以用年份作为地址 考虑关键词的某个线性函数 

这类函数额度有点就是简单 均匀 不会冲突 但是需要率先知道关键词的分布 所以不是非常的常用

#### 数字分析法

抽取关键词的一部分 比如手机号后四位经常用来做身份核验 身份证号 银行卡号当然也可以 也是需要大概了解关键词的特征 并且需要分布均匀

#### 平方取中法

原始关键字平方然后取中间的若干位 比如三位  适合不知道关键词分布 位数不是很大的情况

#### 折叠法

拆分关键词为位数相等的几部分 然后进行求和作为地址 （最后一位不够就短一些） 

只进行一次折叠可能不均匀 或许可以选择从另一端也折叠一次 两个加起来能更均匀一些

适合位数比较大 不知道分布

#### 除留余数法

就是 mod 函数 数学家告诉我们 表长m的时候 一般选择除接近m的最小质数 或者不包含小于20质因子的合数

#### 随机数

随机数一般是伪随机的 里面有一些生成随机数的算法 他可以根据原始值生成对应的随机数 

随机数生成的原理没必要在这里说 其实平方取中也用于随机数生成

### 处理散列冲突

当我们已经发现冲突了 怎么修正？ 当然是有办法的

#### 开放定址法

在遇到冲突的时候 去选择下一个空的散列地址 只要表够大 就不担心找不到

hi(key) = (h(key)+di) mod m

这是核心的开放定址公式 di的不同选取是不同的细分方法     

线性探测：di = i

平方探测：di = ± i2( +12, -12, +22, -22……)  平方探测是为了避免堆积效应的发生 更好的占用整个散列表

随机探测：di是随机数 根据时间或者什么别的生成

#### 再散列函数法

di = i * h2(key)   弄好几个散列函数 一直换函数 问题总能解决

#### 链地址法

地址冲突就冲突 我在这直接扔一个单链表 来一个元素加一个 大不了找到了这个地址再遍历一次进行查找

#### 公共溢出区法

所有地址冲突的 单独找一个溢出区存放 发现哈希不到就来我的溢出区查找 只要冲突数据不多 效果还是很好的

### 散列表的查找

有了前面的哪些思想 这里应该不是问题 

```c
#define m 16	//  哈希表/散列表长度
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
		for (int i = 1; i < m; i++)
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
		for (int i = 1; i < m; i++)
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
	for (int i = 0; i < m; i++)
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
	printf("按散列地址排列：");
	for (int i = 1; i <m; i++)
	{
		printf("%d,", HT[i].key);
	}

	//查找
	int n;
	printf("\n请输入要查找的数：");
	scanf("%d", &n);
	int result = SearchHash(HT, n);
	printf("\n要查找的数在散列表中的地址为：%d  \n", result);
}
```

数组符号实际上只是一个表述 

我们前面所有的Hash都针对数字 因为计算机一切都是二进制表示 只要编码都可以转化为数字 所以Hash针对一切关键字都可以

## 排序

前面我们无数次的提到过一个概念 有序； 在网络上搜索信息我们也经常提到有序的概念 所以引出了非常重要的一类算法 排序问题

### 排序的基本概念和分类

排序的核心就是让序列按照关键码满足非递增或者非递减的关系 一般我们会使用非递减的序列 前面的搜索算法就是根据非递减设计的 很明显的 排序是一种对线性表的操作  

多关键词的排序本质上是单关键词排序的叠加 有时候会采用关键词连缀的方式直接简化 所以后面我们着重进行单关键词的排序

当出现有两个关键词排序条件相等的时候 会引出**排序稳定性**的概念 也就是说 如果有两个人成绩一样 排序之前就在前面的元素应该在排序之后也在前面 就称为稳定 反之称为不稳定 排序算法的稳定性是我们后面需要考虑的一个概念

在前面的研究中我们已经简单了解了数据在内存和外存的差别 在排序的时候也会遇到这个问题 所以区分了**内排序与外排序**的概念 后面只会着重介绍内排序

内排序算法的性能主要采用三个角度衡量 **时间性能 辅助空间 算法复杂性** 这里的复杂性是指算法本身的复杂性 而非时间复杂性

根据排序的主要操作 我们把排序分为 插入排序 交换排序 选择排序 归并排序 四大类

根据算法的复杂度 我们分为   简单算法*含冒泡排序 简单选择排序 直接插入排序*     改良算法 *含 希尔排序 堆排序 归并排序 快速排序*

排序的最基本 需要一个线性表 由于排序经常需要交换元素 所以我们使用一些封装元素  后面不再解释

### 冒泡排序 Bubble Sort

这可以说是最简单的排序算法了 在语言初学阶段就知道他  他的核心思想就是不断两两比较相邻记录的关键词 如果反序就交换 直到没有反序的记录为止 标准的冒泡排序代码如下  时间复杂度$n^{2}$  具有稳定性

```c
void bubble_sort(int a[], int n)   
{
    int i,j,temp;    
    for (j=0;j<n-1;j++)    
    {                           
        for (i=0;i<n-1-j;i++)
        {
            if(a[i]>a[i+1])  
            {
                temp=a[i];      
                a[i]=a[i+1];    
                a[i+1]=temp;
            }
        }
    }    
}
//有时候冒泡排序会做一些无意义的比较 我们可以选择增加flag来避免有序情况下的判断（如果有一轮已经发现没有发生任何交换 终止算法）  
```

### 简单选择排序 Simple Selection Sort

也就是我们最经常提到的选择排序 找到那个最小的放到最前面 然后不断循环 时间复杂度也是$n^2$ 但是性能实际上略微好一点 具稳定性

```c
void select_sort(int R[],int n)    
{
    int i,j,k,index;    
    for(i=0;i<n-1;i++)  
    {
        k=i;
        for(j=i+1;j<n;j++)    
        {
            if(R[j]<R[k])  
                k=j;      
        }
        index=R[i];   
        R[i]=R[k];    
        R[k]=index; 
    }
} 
```

### 直接插入排序 Straight Insertion Sort

插入排序的思路是将一个记录插入已经排好的有序表中 得到新的有序表的过程 前两个元素会在第一轮排序的过程中放好  后面就是选择合适的位置 插进去的过程  时间复杂度$n^{2}$  稳定的

```c
void insertion_sort(int number[],int n)    
{
    int i=0,j=0,temp=0;  
    for(i=1;i<n;i++)  
    {
        temp=number[i]; 
        j=i-1;  
        while(j>=0&&temp<number[j])   
        {
            number[j+1]=number[j];    
            j--; 
        }
        number[j+1]=temp;   
    }              
}
```

### 希尔排序 Shell Sort

在排序算法不断发展的过程中 前面的三种算法和他们的优化方法是很长时间的主流 由于时间复杂度迟迟无法下降 曾有人认为排序算法的时间复杂度不可能低于$n^{2}$ 幸运的是 这个时间复杂度最后被一些科学家突破了 希尔排序就是率先的一批

希尔排序是一种对直接插入排序的优化 他的思路是 竟然时间复杂度以$n^{2}$增加 如果能降低n就能有效降低时间复杂度 所以他选择将原本的序列分组优化为小的子序列 当子序列分别基本有序以后 在对各个序列进行直接插入排序  这里的核心点就在于 **基本有序** 到底是什么 我们来看看代码 

优化子序列是通过间隔k点取一个 分为一组 因此会失去稳定性 跳跃的直接插入能够让序列变得基本有序

```c
void ShellSort(int L[].int n){
    int i,j;
    int increment = n;
    int temp;
    do{
        increment = increment/3+1;
        for(i=increment+1;i<=n;i++){
            if(L[i]<L[i-increment]){
                temp=L[i];
                for(j=i-increment;j>0&&temp<L[j];j-=increment){
                    L[j+increment]=L[j];
                }
                L[j+increment]=temp;
            }
        }
    }
    while(increment>1);
}
//这段代码L[0]位置是空的 不存储数据 很明显 increment的选取非常重要 我们这一的知识一种方法 仅供参考
```

希尔排序算法的核心是间隔选取比较进行跳跃的直接插入  我们比较相隔为increment的元素并且直插排序 在完成一轮dowhile循环后再次缩小increment继续进行 实际上在前面几轮排序完成后 越靠后需要进行的排序工作就越少 这就是希尔排序的核心 在这个方法的优化下 我们将时间复杂度压缩到了 n^1.5 虽然进步不大 但是突破慢速排序很重要  不稳定 平均复杂度会变化 但是不突破nlogn

### 堆排序 Heap Sort

堆排序是对简单选择排序的改进升级 在简单选择排序里面 我们其实在第一次比较完成后又进行了很多次重复的比较 导致时间复杂度的过高 堆排序的思路就是借助堆的新数据结构来辅助 不稳定 平均复杂度$nlogn$极限就是平均

**堆是具有以下特点的完全二叉树 每个结点的值都大于或者等于他的左右孩子的大顶堆 每个结点的值都小于或者等于他的左右孩子的小顶堆**  

在堆的前提下 我们在完全二叉树的部分提到了一个性质

对于一棵有n个结点的完全二叉树 按照层序对结点进行编号 对于任意的结点i

i=1时 他是根节点 没有双亲 i>1的时候双亲是 [i/2]

若2i>n 则结点i没有左孩子（他是叶子结点） 否则i的左孩子是2i

若2i+1>n 则结点i没有右孩子 否则i的右孩子是2i+1

很明显的 大顶堆和小顶堆采用层序遍历的话就是一个大致有序数组 而所谓的堆排序 就是把原始序列构造成一个大顶堆 把堆顶元素移动到末尾 再让剩下的元素构成大顶堆 不断重复 就能得到有序的序列 下面我们疑惑的问题可以靠代码理解

```c
void swap(int* a, int* b) {
    int temp = *b;
    *b = *a;
    *a = temp;
}//懂得都懂 后面那么多结点交换 这样轻松一些
void max_heapify(int arr[], int start, int end) {
    //建立父节点指标和子节点指标
    int dad = start;
    int son = dad * 2 + 1;
    while (son <= end) { //若子节点指标在范围内才做比较
        if (son + 1 <= end && arr[son] < arr[son + 1]) //先比较两个子节点大小，选择最大的
            son++;
        if (arr[dad] > arr[son]) //如果父节点大于子节点代表调整完毕，直接跳出函数
            return;
        else { //否则交换父子内容再继续子节点和孙节点比较
            swap(&arr[dad], &arr[son]);
            dad = son;
            son = dad * 2 + 1;
        }
    }
}
void heap_sort(int arr[], int len) {
    int i;
    //初始化，i从最后一个父节点开始调整
    for (i = len / 2 - 1; i >= 0; i--)
        max_heapify(arr, i, len - 1);
    //先将第一个元素和已排好元素前一位做交换，再从新调整，直到排序完毕
    for (i = len - 1; i > 0; i--) {
        swap(&arr[0], &arr[i]);
        max_heapify(arr, 0, i - 1);
    }
}
//修正完以后的二叉树进行层序遍历就可以有序了
```

### 归并排序 Merging Sort

归并就是合并的意思 他的核心是把多个有序的表合成一个新的有序表 如果初始有n个记录 就视为n个有序的子序列 长度都是1 然后两两归并为长度[n/2]的长度为1或2的子序列 如此重复继续归并 最后得到一个长度为n的有序序列 这被称为2路归并排序 也是我们在这里介绍的一种  它稳定 平均复杂度$nlogn$ 极限就是平均 下面直接看代码帮助我们理解

在归并的时候实际上是用用两个指针分别追踪两个小数组 然后单独开辟空间存放归并的结果

```c
void merge_sort_recursive(int arr[], int reg[], int start, int end) {
    if (start >= end)
        return;
    int len = end - start, mid = (len >> 1) + start;
    int start1 = start, end1 = mid;
    int start2 = mid + 1, end2 = end;
    merge_sort_recursive(arr, reg, start1, end1);
    merge_sort_recursive(arr, reg, start2, end2);
    int k = start;
    while (start1 <= end1 && start2 <= end2)
        reg[k++] = arr[start1] < arr[start2] ? arr[start1++] : arr[start2++];
    while (start1 <= end1)
        reg[k++] = arr[start1++];
    while (start2 <= end2)
        reg[k++] = arr[start2++];  //这三个while循环就是对拆分双指针递归的实现 
    								//理解一下思路 我们从两个数组里从两边的头开始比大小 找小的塞进reg里面 然后下一位
    for (k = start; k <= end; k++)
        arr[k] = reg[k]; //把reg临时存放的数据扔回去方便递归回去调用
}
void merge_sort(int arr[], const int len) {
    int reg[len];
    merge_sort_recursive(arr, reg, 0, len - 1);
}
//这里只是进行了一次调用 方便我们进行前面函数的递归操作
```

### 计数排序 Counting Sort
计数排序(Counting Sort)不是基于比较的排序算法，

其核心在于将输入的数据值转化为键存储在额外开辟的数组空间中。 作为一种线性时间复杂度的排序，计数排序要求输入的数据必须是有确定范围的整数。它的基本思想是：给定的输入序列中的每一个元素x，确定该序列中值小于等于x元素的个数，然后将x直接存放到最终的排序序列的正确位置上。

### 桶排序 Bucket Sort
桶排序（Bucket sort）或所谓的箱排序，是一个排序算法，工作的原理是将数组分到有限数量的桶里。每个桶再个别排序（有可能再使用别的排序算法或是以递归方式继续使用桶排序进行排序），最后依次把各个桶中的记录列出来记得到有序序列。

### 基数排序 Radix Sort
基数排序（Radix sort）是一种非比较型整数排序算法。

原理是将整数按位数切割成不同的数字，然后按每个位数分别比较。基数排序的方式可以采用LSD（Least significant digital）或MSD（Most significant digital），LSD的排序方式由键值的最右边开始，而MSD则相反，由键值的最左边开始。

- **MSD**：先从高位开始进行排序，在每个关键字上，可采用计数排序
- **LSD**：先从低位开始进行排序，在每个关键字上，可采用桶排序

### 快速排序 Quick Sort

快速排序是我们在前面提到的最基础的排序方法 冒泡排序的升级 也是通过不断地比较和移动 不过他增大了比较和移动的距离 从而减少了比较和交换的次数

基本思想 ：通过一套排序把待排记录分割成独立的两部分 一部分的关键词均比另一部分要小 从而分别对着两部分进行排序 从而让整个数列有序 他虽然看起来叙述向希尔排序 但是实际上和原本的基本更大不一样 我们直接看代码‘

```c
void QuickSort(Sqlist *L){
    Qsort(L,1,L->length);
}
//和归并一样 因为涉及到递归调用的问题我们添加了一个封装层 
void Qsort(Sqlist *L,int low,int high){
    int pivot;
    if(low<high){
        pivot=Partition(L,low,high); //用调用了一个函数 他的作用是选择一个关键词 是谁无所谓 
        							//然后找到一个位置让他左边都比他小 右边都比他大
        Qsort(L,low,pivot-1);
        Qsort(L,pivot+1,high);//两次递归调用
    }
}
int Partition(Sqlist *L,int low,int high){
    int pivotkey;
    pivotkey = L->r[low];
    while(low<high){
        while(low<high&&L->r[high]>=pivotkey){
            high--;
        }
        swap(L,low,high); //就当这是一个封装好的函数就行 虽然C里没有 这里我们的核心是理解算法
        while(low<high&&L->r[low]<=pivotkey){
            low++;
        }
        swap(L,low,high); //就是从两端找元素 找到了就和选定的pivot交换 最后形成一个左小右大 high=low的时候OK了
    }
    return lowl;
}
//1 开始存元素的地方 别用前面那数组了 还是链表好用 
```

快速排序并不稳定 其时间复杂度一般为 $nlogn$  最坏情况下（原本的数列就有顺序）时间复杂度是$n^{2}$

**优化pivot**

快速排序的算法性能很明显的被我们选取的pivot的值所影响 这个值越接近整体的中位数 后面的算法就会算的更少 所以我们会引入三数字取中 九数字取中的方式 希望它更接近中间值的关键字

**优化交换**

舍弃封装好的swap函数 改为替换 希望在这里节省一些操作 但是舍弃封装 会更接近底层 

**优化小数组方案**

小数组时快速排序并不快速 在长度低于几十（很主观）的时候 选择直接插入排序就好了

**优化递归**

递归对计算机性能的消耗是不小的 所以存在尾递归的优化方式来提高性能

### 结尾语

很有趣的是 这个排序算法被称为快速排序 这个命名其实就能说明很大的问题 如果有更好的排序算法 他就不名副其实了 实际上快速排序经过不断地演进后 就是目前整体效率最高的算法 很难继续被优化了
