---
title: 'Introduction to Data Structures: Linear Lists, Trees, Graphs, and Search'
title_zh: 数据结构导论：线性表、树、图与查找排序
date: 2025-05-12 22:24:30 +0800
categories:
- Programming
- CS Foundations
tags:
- Data Structures
- Algorithms
author: Hyacehila
mathjax: false
hidden: true
excerpt: Introduces data structures through core concepts, linear lists, stacks, queues, trees, graphs, search, sorting, and
  programming fundamentals.
description: Introduces data structures through core concepts, linear lists, stacks, queues, trees, graphs, search, sorting,
  and programming fundamentals.
excerpt_zh: 整理数据结构基本概念、线性表、栈、队列、树、图、查找和排序等程序设计基础。
permalink: /blog/2025/05/12/data-structures-introduction/
lang: en
translation_key: 2025-05-12-data-structures-introduction
translation_status: machine
translation_source_hash: 8d325d1e59bda8b44edb63b7bde5f44d9222c421398d23cb93fdc94537e75a16
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>In this part of the study, we learn the basics of data structure and algorithms, but the more complex algorithms will have to be learned in other teaching materials, but it is an important basic course in program design and it is not difficult.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2024/12/18/search-and-sorting-algorithms/">Find and Sort Algorithms: Retrieval, Dispersed Lists and Sort</a>、<a href="/en/blog/2025/05/13/algorithm-design-and-analysis/">Algorithm design and analysis: partitioning, dynamic planning and algorithms</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h2>Data structure draft</h2>
<p>In the early computer development process, numerical calculations are a problem that it needs to address, so they are often used in dealing with problems.<strong>Abstract model, design algorithm, program preparation</strong>The process; but the non-value problem is emerging, and the shift in our thinking here;<strong>Data structure is the discipline of the program design computer that studies non-value calculations, the subject of the operation and the relationship.</strong> After that, the thinking of dealing with the problem was to<strong>Select the structure type Design algorithm</strong>The process of this... <strong>Program design = data structure + algorithm</strong></p>
<h3>Basic concepts and terminology</h3>
<h4>Data</h4>
<p>Data are symbols that describe objective things, are actionable objects in computers, are symbols that can be identified by computers and entered into the computer-processing collections, whether numerical data or non-numeric coded data, are data that we're looking at here.</p>
<p>Data elements are the basic units that make up data, and in computers we tend to treat them as a whole.</p>
<p>The data items are the smallest units of data indissoluble, the composition of data elements (attracts to analog objects).</p>
<p>Data audience is a collection of data elements of the same nature (e.g., grouping multiple objects created by the same category)</p>
<p>The data structure is the collection of connecting data elements, because the data elements are not isolated, and we're not going to consider their connection until we design the program.</p>
<h4>Logical and physical structures</h4>
<p><strong>Logical structure indicates the interrelationship of data elements in the data object</strong> This is one of the things we're gonna really take seriously.</p>
<ul>
<li>Pool structure: equality of all elements</li>
<li>Linear structure: there's a one-to-one relationship between data elements A to B to C and this </li>
<li>Tree structure: a multi-layered relationship</li>
<li>Graphical structure: multi-to-multi-relationship</li>
</ul>
<p>We usually use a diagram to indicate their connection, which is basically numerical to the graphics, and each node represents a data element, and the connection, if not arrows, is a two-way, forward and subsequent element is important. </p>
<p><strong>Physical structure is the storage structure of computers.</strong> </p>
<ul>
<li>The sequence structure, the array in the C language is like this.</li>
<li>The chain structure, the chain watch in the C language.</li>
</ul>
<p>Logical structure is problem-oriented, physical structure is computer-oriented, and it's important to get a bridge between them.</p>
<h4>Abstract Data Type</h4>
<p><strong>Data type refers to the sum of a set of values and the total number of operations defined on this set</strong></p>
<p>The data type is classified to determine the space to be occupied according to needs, and it is a means of saving computer resources. </p>
<ul>
<li>Atomic type: Non-deseparable basic type Integers Solid Characters, Characters, etc.</li>
<li>Structure type: Multiple atom types combined, for example, number Group</li>
</ul>
<p>The abstract data type is an abstract feature of the existing data type, which is designed to study the universal nature of things, and we use some of the conventional data types, and we study the types of data that we own, based on our own needs for use, such as structure, dictionary types.</p>
<p>ADT should include the name of the data type, the definition of the data element and the logical relationship, the description of the operation and the results of the operation, and the code of achievement of ADT is what we're always dealing with behind us.</p>
<h2>Algorithm</h2>
<p>Algorithms and data structures are two elements that cannot be separated, but algorithms are too complex, and we learn more from the possibility of deeper.</p>
<h3>Algorithms definition and characteristics</h3>
<p>Algorithms are a description of the steps to solve a particular problem, and in computers they are command sequences; obviously, there is no universal algorithm in the world, and that's why he's complicated.</p>
<h4>Input Output</h4>
<p>It's very understandable that algorithms can be uninputable, but they basically have output, and no output algorithms are meaningless.</p>
<h4>Poor.</h4>
<p>Algorithms should not be infinity.</p>
<h4>Determination</h4>
<p>The algorithm is precise. Every step is fixed.</p>
<h4>Feasibility</h4>
<p>If your algorithm is so complicated that it's completely impossible to achieve, then it's pointless.</p>
<h3>Design requirements and efficiency measures</h3>
<p>Algorithms are not the only ones, so we'll have some requirements for these different algorithms and a measure of their good or bad.</p>
<h4>Correctness</h4>
<p>The input, output, processing, and so on, it's not a question of being able to handle the problem correctly, and get the right answer to the question, and the level is as follows, and we can't always guarantee that one algorithm is correct, so step three is enough.</p>
<ul>
<li>Ungrammatical error</li>
<li>Legitimate input returns the correct result</li>
<li>Illegal input gives appropriate hints</li>
<li>For intentional test data sets, it also returns correctly.</li>
</ul>
<h4>Readability</h4>
<p>If anyone else doesn't understand your algorithm, it's hard for him to continue to advance.</p>
<h4>-Staffy.</h4>
<p>The proper handling of illegal data does not produce unusual and inexplicable results.</p>
<h4>Time-efficient and low storage</h4>
<p>It's the expression of the complexity of time and space.</p>
<h4>After-action statistical methodology</h4>
<p>The time complexity of algorithms is measured by the means of the production of test data sets, running programs, but it takes a lot of effort to test them, which is vulnerable to the condition of the equipment, so he is rarely used to measure algorithms.</p>
<h4>Prior estimation methodology</h4>
<p>First, we need to understand the scale of the problem input, which is typically expressed in n, which is the same size of the two algorithms that are designed for the same purpose, and the number of basic operations that make up the function of the time complexity that we want to study, and the amount of computing resources that are consumed by the algorithms with different complexity of time, which is growing as the scale increases, and eventually creates a qualitative gap.</p>
<h3>The function 's incremental growth and complexity of time degrees</h3>
<p>How can the algorithm be directly evaluated for its good or badness based on its time-complexity function? We're going to give you some complementary conclusions.</p>
<p><strong>Progressive growth of functions</strong> If it's not...&gt;After N, one function is always bigger than the other, which is called his growth is growing faster than the other. </p>
<p>And at this point we can get some of the supporting conclusions, which are the basic ones of mathematical analysis.</p>
<ul>
<li>Adding constants is negligible</li>
<li>Multiplier of the highest sub-point to ignore</li>
<li>Non-highest sub-points to ignore</li>
<li>The highest sub-point is very important.</li>
</ul>
<p>In fact, we're just looking at the size of the step, which is the complexity of the time we need to focus on, which is generally called the O-class, and the O-class study, and one mathematician should be able to easily think that there's no O-12, but O (lgn) O (lnn) O (m*n) O (nlgn) O (n) O (n) (n) (n) (n) (n) (n) (n) (n) (n)) (i) (i)) (i)) (ii) that the relationship between them is not an infinity of the size of the relationship, it should be not a problem for the mathematicals.</p>
<h3>Worst and average</h3>
<p>Average and worst means something, but better still, it's not. </p>
<p>The worst is a guarantee that nothing worse will happen, so we're always looking at him as a measure.</p>
<p>On average, it's the best of all, but we can't do this with technology.</p>
<h3>Space complexity</h3>
<p>We often have low space complexity, and we often use space-for-time techniques in algorithm design, and after all, users don't necessarily say what they want because they're using more storage, but they're evaluating the cartons they encounter, and they're going to get some of the usual content calculated, and direct calls are a good way to use it when needed.</p>
<p>It's a programr that should be taking into account when designing the bottom of the program, but it doesn't seem important to a developer.</p>
<p>The SSI O(1) means that all operations do not consume additional levels of memory, and the higher complexity of the space is the same.</p>
<h2>Linear Table</h2>
<p>The linear table is one of the simplest and most commonly used data structures, and he has a head, a tail, a sequence;</p>
<p>Linear table:<strong>Limited series of zero or more data elements</strong></p>
<p>The sequence, which means that there is order between elements, and the limited is that the computer's characteristics are only mathematically capable of processing infinite sequences.</p>
<h3>Abstract data type of linear table</h3>
<ul>
<li>Create</li>
<li>Initialize</li>
<li>Reset to Empty</li>
<li>Visits conducted on an ongoing basis by means of a sequence</li>
<li>Find</li>
<li>Total length of visits</li>
<li>Insert and Delete Data</li>
</ul>
<p>We think these operations are important at this point, and we can add new ones at any time, but we might as well consider these basic operations as linear tables.</p>
<p>As for the properties of the linear tables, it's only possible to understand the concept of linear tables, and the most important is the limited series.</p>
<h3>Order storage structure</h3>
<p>We're thinking first of all of using sequenced storage structures to store this linear table, which is to find a continuous memory to store, because of the characteristics of the linear tables, it's a pretty good way to use a 1-dimensional array, so this is the first line of code we're writing in the data structure class.</p>
<pre><code class="language-c">#define MAXSIZE 20
typedef int ElemType;
typedef struct
{
    ElemType data[MAXSIZE];
    int length;
}Sqlist;
//存储的类型是自由变化的 是结构也无所谓 这个我们这里强调一次 后面就不提的 很明显的我们选择使用这个结构作为存储数据的形式
//起始位置 数组data 最大容量 长度 均在我们的线性表中被体现的出来
</code></pre>
<p>The length of arrays is different from the length of linear tables, which are fixed when VLA is not introduced, and then run with the code, which is linked to the length of linear tables and the number of elements stored.</p>
<p>We're not going to repeat here about the address.</p>
<pre><code class="language-c">#define OK 1
#define ERROR 0
#define TRUE 1
#define FALSE 0
#define int Status
Status GetElem (Sqlist L,int i,ElemType *e)
{
    if(L.length==0 || i&lt;1 || i&gt;L.length){
        return ERROR;
    }
    *e=L.data[i-1];
    return OK;
}
//建立访问线性表中元素的函数 由于不需要修改所以选择值传递
Status ListInsert(Sqlist *L,int i,ElemType *e)
{
    int k;
    if(L-&gt;length==MAXSIZE){
        return ERROR;
    }
    if(i&lt;1||i&gt;L-&gt;length+1){
        return ERROR;
    }
    if(i&lt;=L-&gt;length){
        for(k=L-length-1;k&gt;=i-1;k--){
            L-&gt;data[k+1]=L-&gt;data[k];
        }
    }
    L-&gt;data[i-1]=e;
    L-&gt;length++;
    return OK;
}
//检验插入位置后 插入元素到原本的线性表里面 这个程序的设计拒绝我们把元素跳跃的插入 否则线性表的前后元素就会出问题
Status ListDel(Sqlist *L,int i,ElemType *e)
{
    int k;
    if(L-&gt;length==0){
        return ERROR;
    }
    if(i&lt;1||i&gt;L-&gt;length){
        return ERROR;
    }
    *e = L-&gt;data[i-1] //返回一下被删掉的元素
    if(i&lt;=L-&gt;length){
        for(k=i;k&lt;L-&gt;length;k++){
            L-&gt;data[k-1]=L-&gt;data[k];
        }
    }
    L-&gt;length--;
    return OK;
}
//删除我们不需要的元素并且给出了被删除的元素
//如果要研究这几个函数的时间复杂度的话 查询算法复杂度1 后面的平均算法复杂度是n（研究平均才有意义）
//设计其他的函数其实并不是一个复杂的事情 在这里我们就留作练习了 考虑的细致周到一点就好
//整表的创建就是建立一个结构初始化后并且填入我们的初始数据 清为空就是控制长度为0 你不需要告诉我们存储了什么 只要知道哪些数据是无效的就好了 删除就是释放所有占用的内存 都不困难
</code></pre>
<h3>Chained storage structure</h3>
<p>The emergence of chain storage structures is very important for data structure, and many of the logical structures that follow are difficult or impossible to store in sequence, and the core of chain storage is the random use of unoccupied memory, and the savings in memory are obvious.</p>
<p>We call the whole of the pointer field and the data field a node, Node, and obviously we're only going to use the pointer field as one. </p>
<p>The position of the first element of the single-chain table is called the head pointer, and the last element pointer is directed at the NULL, and sometimes we build it.<strong>Headpoint storage of information on the header and some information on the table of chains at large</strong> Or simply headpoint is a pointer to the next element** if the linear table is empty, the headpoint is to the NULL</p>
<pre><code class="language-c">typedef struct Node
{
    ElemType data;
    struct Node *next;
}Node;//结点建立

typedef struct Node *LinkList; //链表建立 这一步完全可以浓缩到上一步 在栈里面我们就能看到这个写法 这里其实我们没有准备头节点来存储一些信息
// 这就是最基本的节点的构成 由指针域和数据域 至于数据域究竟是什么样子我们这里不重视
Status GetElem(LinkList L,int i,ElemType *e){
    int j;
    LinkList p;
    p = L-&gt;next;
    j=1;
    while(p&amp;&amp;j&lt;i){
        p = p-&gt;next;
        ++j;
    }
    if(!p||j&gt;1){
        return ERROR;
    }
    *e = p-&gt;data;
    return OK;
}
//顺序访问我们的单链表 知道找到元素 很明显在这个查找时间复杂度是n优势不明显 由于单链表没有控制表长 所以我们不方便使用for循环
Status ListInsert(LinkList *L,int i,ElemType e)
{
	int j;
    LinkList p,s;
    p = *L;
    j=1;
    while(p&amp;&amp;j&lt;i){
       p = p-&gt;next;
       ++j;
    }
    if(!p||j&gt;1){
        return ERROR;
    }
    s = (LinkList)malloc(sizeof(Node));
    s-&gt;data = e;
    s-&gt;next = p-&gt;next;
    p-&gt;next = s;
    return OK;
}
//这是插入单链表的方法 其本质并不复杂 采用malloc申请内存是为了处理C语言的变量生存期问题
Status ListDel(LinkList *L,int i,ElemType *e)
{
	int j;
    LinkList p,s;
    p = *L;
    j=1;
    while(p-&gt;next&amp;&amp;j&lt;i){
       p = p-&gt;next;
       ++j;
    }
    if(!(p-&gt;next)||j&gt;1){
        return ERROR;
    }
	q = p-&gt;next;
    p-&gt;next = q-&gt;next;
    *e = q-&gt;data;
    free(q);
    return OK;
}
//这是删除 我们用free释放空间 请在数据结构里面避免指针运算 因为这是C语言特性 数据不只在C语言使用
//明显的我们能看到 链式存储结构在删除和插入有时间复杂度的优势
</code></pre>
<p><strong>Create and delete the entire table</strong> We should consider the method of deletion and creation in the chain structure alone.</p>
<pre><code class="language-c">//单链表的整表创建就是循环建立元素节点并且依次连接形成的
void CreateList(LinkList *L,int n){
    LinkList p;
    int i;
    *L = (LinkList)malloc(sizeof(Node));
    (*L)-&gt;next = NULL;
    for(i=0;i&lt;100;i++){
        p =  (LinkList)malloc(sizeof(Node));
        p -&gt;data = 1;
        p-&gt;next = (*L)-&gt;next;
        (*L)-&gt;next = p;
    }
}
//这是头插的结构 新的节点一直在头指针L和第一个元素之间 不断补充新的第一个元素
void CreateList(LinkList *L,int n){
    LinkList p;
    int i;
    *L = (LinkList)malloc(sizeof(Node));
    r = *L;  //结尾节点设置
    for(i=0;i&lt;100;i++){
        p =  (LinkList)malloc(sizeof(Node));
        p -&gt;data = 1;
        r-&gt;next = p;
        r=p;
    }
    r-&gt;next = NULL;
}
//这就是尾插的方法 新的节点一直在旧的节点的尾部
//如果我们已经完成了整表的创建想要添加新的元素直接用Insert的方法就可以 这里只是从零创建需要的
Status ClearList(LinkList *L){
    LinkList p,q;
    p = (*L)-&gt;next; //第一个结点
    while(p){
        q=p-&gt;next;
        free(p);
        p=q;
    }
    (*L)-&gt;next = NULL; //头结点的释放
    return OK;
}
//这是对整个单链表的释放 我们挨个存储下一个节点并且释放上一个节点 最后让头结点指针域归NULL
</code></pre>
<p>And that's the time when the speech about the single-chain table is here, and it's easy to see the great advantage of chain-storage structures that are easily inserted and deleted, and the sequence-storage structures are more accessible and limited in length, and then we'll see that an interesting chain-watch structure is linked to the single-chain table, and in fact many of the complex data structures behind it are the simplest structures to gradually increase complexity.</p>
<h4>Static Chain Table</h4>
<p>The static chain is a very special product, and after the introduction of the important memory operation of the pointer in the C language, the next advanced language, such as Python and Java, is a system of object references (i.e. object-oriented language). So they can also achieve data structure in a relatively easy way, but earlier language does not have these techniques, so some programmers choose to use arrays to achieve some degree of single-chain function, that is, static-chain, but he is a stopgap device that loses the important advantages of chain storage structures -- - Flexible use of memory</p>
<h4>Cyclical Chain</h4>
<p>The biggest problem with the single-chain table is that it can only be accessed from the top, but the loop-link can solve it to a certain extent. </p>
<p>The circle table is very simple to form, but the pointer field for the terminal node is changed from an empty pointer to a pointer, so the whole chain table forms a ring, and he's a unique example of a single chain table. </p>
<p>In the circulation chain, to facilitate the design of more codes, the nodes are basically there, and the rules have been formed. </p>
<p>The biggest difference between the cycling and single-chain tables is that the conditions for judging the cycle are different.</p>
<p>Occasionally, we introduce end-pointers to facilitate access to elements, but this is not common.</p>
<h4>Two-way Chain</h4>
<p>Naturally, we'll introduce chains that can look at both the back element and the front element so that we can overcome the one-way disadvantage of expanding the pointer field.</p>
<pre><code class="language-c">typedef struct DulNode
{
    ElemType data;
    struct DulNode *prior;
    struct DulNode *next;
}DulNode,*DulLinkList;
//写出一个双向链表的结点并不困难
//其他的函数其实也非常好写 只是更加繁琐了 细致细心 对齐各个指针就可以
</code></pre>
<h2>Inn and queue</h2>
<p>We're actually working on two special linear forms here.</p>
<p><strong>Stack is the linear limit for insertion and deletion at the end of the table Table</strong></p>
<p>We call the one that allows insertion and deletion operations the toptop, and the other the bottom bottom, which is an empty vault, which does not contain any elements, and the vault is actually a back-to-back structure, a structure called LIFO, and we need to understand that the core of the stack is that the stack is a special linear table, and he operates from the end of the table or from the top of the stack.</p>
<p>Obviously, we think the ADT and the linear table should be much different, but the del and insert function should be replaced by the special features of the push and pop.</p>
<h3>Order storage structure</h3>
<pre><code class="language-c">typedef struct{
    ElemType data[MAXSIZE];
    int top; //记录栈顶的位置 top=0 代表元素只有一个 top为-1意味着栈是空的
}Sqstack;
//建立顺序存储结构的栈
Status Push(SqStack *S,ElemType e){
    if(S-&gt;top == MAXSIZE-1){
        return ERROR;//满栈
    }
    S-&gt;top++;
    S-&gt;data[S-&gt;top]=e;
    return OK;
}
//这是压栈的函数
Status Pop(SaStack *S,ElemType *e){
    if(S-&gt;top == -1){
        return ERROR;//空栈
    }
    *e = S-&gt;data[S-&gt;top];
    S-&gt;top--;
}
//这是出栈的函数 实际上只要借助top的位置就知道存到哪里了 不需要单独的初始化操作 这是我们现在的代码书写比较特殊的一点
</code></pre>
<p>We're not gonna write any more here.</p>
<h4>Two stacks share storage space</h4>
<p>Sometimes we'll meet two data structures of the same type, and they have a relationship that we'll use to save space. It's just a storage technique, not that we have to use it, just a few small examples, without detailing it.</p>
<pre><code class="language-c">typedef sturct{
    ElemType data[MAXSIZE];
    int top1;
    int top2;
}SqDoubleStack;//增加指针的数量
Status Push(SqDoubleStack *S,ElemType e,int StackNumber){
    if(S-&gt;top1+1==S-&gt;top2){
        return ERROR;
    }
    if(StackNumber==1){
        S-&gt;data[++S-&gt;top1]=e;
    }
    else if(StackNumber==2){
        S-&gt;data[--S-&gt;top2]=e;
    }
    return OK;
}
//基本上就是这样的思路 额外增加一个选择你进什么栈的选择 出栈原理也是完全一样的 实际书写代码的时候完全可以把修正栈顶和赋值语句分开分布
</code></pre>
<h3>Chained storage structure</h3>
<p>And because of the single-chain structure, the structure of the bar also needs a knob to meet our access needs, and it's easy to understand whether to combine them in one place, and at this point we find that the head nodes are really useless and that the determination of the void is entirely based on the NULL.</p>
<pre><code class="language-c">typedef struct StackNode{
    ElemType data;
    struct StackNode *next;
}StackNode,*LinkStackPtr;//结点建立

typedef struct LinkStack{
    LinkStackPtr top;
    int count;
}LinkStack;//链栈建立 和前面有微小的不同 整体非常接近
//在实际的链式结构操作上 栈和链表也有很大的接近之处 只要pop和push函数需要考虑重写 链栈不得不考虑头节点 因为count数据要单独的进行存储
Status Push(LinkStack *S,ElemType e){
    LinkStackPtr s = (LinkStackPtr)malloc(sizeof(StackNode));
    s-&gt;data = e;
    s-&gt;next = S-&gt;top;
    S-&gt;top = s;
    S-&gt;count++;
    return OK;
}
//实际上原理非常的简单 我们最重要的是理解这里指针的情况就可以解决
Status Pop(LinkStack *S,ElemType *e){
	*e = S-&gt;top-&gt;data;
    if(StackEmpty(*S)){
        return ERROR;
    }
    LinkStackPtr s = S-&gt;top
    S-&gt;top = S-&gt;top-&gt;next;
    free(s);
    S-&gt;count--;
    return OK;
}
//弹出栈的函数其实也很好写 其实最难的是C语言指针比较复杂的规则
</code></pre>
<p>The inn has been sealed in many advanced languages.</p>
<h3>Use of the stack</h3>
<p>The function is a more common program design idea, and it is easy to observe that when one layer of the function is re-entry, they go back to the last operation, and that's how the idea is, and then the data that is being squeezed into the stack requires the first exit to be involved in the operation, and of course, now that the advanced language is automatically managed, we can just re-enter the function.</p>
<p><strong>The queue is also a special linear table that allows insertion at one end and deletion at the other.</strong> It means that the queue is a data structure for First In First Out, which allows for the insertion of what is called a team head, and for the deletion of what is called a linetail, which is actually a very well-established pattern of our lives, so it's very broad, like the keyboard input of data into the system, which is the first entry system. </p>
<h3>Ordered Storage & Cycle Queue</h3>
<p>The queue for designing a sequenced storage structure should not be a problem, considering that we just need to add a new element to the end of the array when we're in line, to send the first element out of line and to add to the rest of the queue, but is the time to get out too complicated? </p>
<p>Can we use a little bit of a special technique, like changing the position of the team leader, which can be effective in reducing the complexity of time, of course, by having two fingers at the end of the team? A needle.</p>
<p>But it's gonna be a new problem, and the team and the tail pointer are empty, but the full set is not a good idea, the end pointer is not always full at the end, and there's probably room ahead, and that's the fake spill.</p>
<p>The way the loop queue gives us is if we find out that the end pointer is rear, we start to add data to the array until the front of the rear chase, and then there's another problem, and the heavy contract is full and empty, and the idea of handling it is simple, and if we find out that rear is on the way to catch the front, we leave an empty slot and report directly to the full queue.</p>
<p>The code is as follows:</p>
<pre><code class="language-c">typedef struct{
    ElemType[MAXSIZE];
    int front;
    int rear; //指向第一个空的尾而不是有元素的 根据不同的习惯 记得修正自己的代码
}SqQueue;
Status InitQueue(SqQueue *Q){
    Q-&gt;front=0;
    Q-&gt;rear=0;
    return OK;
}
int QueueLength(SqQueue Q){
    return (Q.rear-Q.front+MAXSIZE)%MAXSIZE
}
Status EnQueue(SqQueue *Q,ElemType e){
    if((Q-&gt;rear+1)%MAXSIZE==Q-&gt;front){
        return ERROR;
    }
    Q-&gt;data[Q-&gt;rear]=e;
    Q-&gt;rear = (Q-&gt;rear+1)%MAXSIZE;
    return OK;
}
Status DeQueue(SqQueue *Q,ElemType *e){
    if(Q-&gt;front == Q-&gt;rear){
        return ERROR;
    }
    *e=Q-&gt;data[Q-&gt;front];
    Q-&gt;front = (Q-&gt;front+1)%MAXSIZE;
    return OK;
}
//虽然说循环队列是一个非常优秀的存储方式 但是数组真正溢出仍然无法不免 链表结构才是数据结构学科的真正核心
</code></pre>
<h3>Chained storage structure</h3>
<p>We still follow the usual practice of pointing the nodes off the chain and the important designs that are useful behind it, so that the front points to the nodes rather than the real team leader, rear points to the current team tail.</p>
<pre><code class="language-c">typedef struct QNode{
    ElemType data;
    struct QNode *next;
}QNode,*QueuePtr;
typedef struct{
    QueuePtr front,rear;
}LinkQueue;
//还是我们的管理一套 把结点 结点指针 链队列分开进行结构命名 实现更舒服的存储效果 我们使用起来也会更加顺畅
Status EnQueue(LinkQueue *Q,ElemType e){
    QueuePtr s = (QueuePtr)malloc(sizeof(QNode));
    if(!s){
        exit(OVERFLOW);
    }
    s-&gt;data = e;
    s-&gt;next = NULL;
    Q-&gt;rear-&gt;next = s;
    Q-&gt;rear = s;
    return OK;
}
Status DeQueue(LinkQueue *Q,ElemType *e){
    QueuePtr p;
    if(Q-&gt;front==Q-&gt;rear){
        return ERROR;
    }
    p = Q-&gt;front-&gt;next;
    *e = p-&gt;data;
    Q-&gt;front-&gt;next = p-&gt;next;
    if(Q-&gt;rear == p){
        Q-&gt;rear = Q-&gt;front;
    }
    free(p);
    return OK;
}
//没有什么难以理解的地方
</code></pre>
<h2>Thread</h2>
<p>And what we're actually looking at in this chapter is the type of data that's this kind of string, which is obviously a very interesting type of data because it's not just a digital thing that people have developed since modern computer technology, but a character-spectrum string that's stored in character arrays, and it's clear that we've started to study character processing in a very, very detailed way from here, and then we've developed in more advanced languages, where we've just introduced something that is not complicated, and more of the content has been sealed in advanced languages, and we don't need to get that close to them.</p>
<p>The length of the string, the empty string, the empty string, the substring and the main string, the position of the lead string, which we already know enough about in the base of the C language, and we don't repeat it here.</p>
<p><strong>Thread comparison</strong></p>
<p>The size of the two numbers can no longer be easy, but the string is also large, and we can handle the problem well in English only, and the first 256 characters of the further Unicode are the same size as the ASCII code, which is the size of the ASCII code in the next character comparison, and we'll use the understanding string later.</p>
<h3>Stringed ADT and Storage</h3>
<p>The same thing, but the basic operation of the string and the linear appearance are different, and we're concerned about the existence of the substring, but the linear table does not have this concept, because the advanced language has already sealed most of the operation of the string, so the following is just some introductory material.</p>
<p>The string is usually stored in sequence, chain storage has no advantage for the string, actually, the String type data is stored in a stack, the system is distributing him dynamically, and stacks can be managed with a maloc and free.</p>
<pre><code class="language-c">int Index(String S,Strint T,int pos){
    int n,m,i;
    String sub;
    if(pos&gt;0){
        n= StrLength(S);
        m= StlLength(T);
        i = pos;
        while(i&lt;=n-m+1){
            SubString(sub,S,i,m);
            if(StrCompare(sub,T)!=0){
                i++;
            }
            else{
                return i;
            }
        }
    }
    return 0;
}
//这里我们借助一些基本函数实现了一个查找的函数 后面会研究一些独立于这些函数的方法 这里只是让我们知道目前的高级语言关于串的程序是怎么设计的
</code></pre>
<h3>PARK Mode Matching Algorithms with KMP Mode</h3>
<p>Both algorithms deal with the existence and location of the substring. <strong>A simple pattern matching algorithm</strong>And the next is code expression.</p>
<pre><code class="language-c">//请注意 字符串的第一位也就是0出存储了字符串的长度
int Index(String S,String T,int pos){
    int i = pos;
    int j = 1;
    while(i&lt;=S[0]&amp;&amp;j&lt;=T[0]){
        if(S[i]==T[j]){
            j++;
            i++;
        }
        else{
            i = i-j+2;
            j = 1;
        }
    }
    if(j&gt;T[0]){
        return i-T[0];
    }
    else{
        return 0;
    }
}
//在这个程序设计里面 我们就是最简单的挨个核对主串的每一个子串看看能不能找到完全符合的
//但是很容易发现 如果原本的字符串里面有一些部分重合的段 这个匹配算法的时间复杂度是非常高的 尤其是原本的比对会被转换成二进制码的时候
</code></pre>
<p><strong>So we need to improve the KMP model matching algorithm.</strong>  Algorithms are named by the acronym of the algorithm developer's name.</p>
<p>We need to start with the algorithms, because he's not very straight. Watch</p>
<p>KMP algorithms and cores are not consistent with the initials and the back of the original substring T, so if you complete some of the same judgments, you can discard some unnecessary judgements, namely that the current i value of the main string does not need to continue to increase according to 12345, and that the backtracking process of i values can be bypassed, save resources** (the value of core point 1 i does not need to be traced, and it can be directly followed without problems)**</p>
<p>And what about the j-value, which is obviously important is that when there's no repetition, j returns to one to be complete, but if there's repetition, j will have a different change, which is related to the suffixity of the string before the current character, and we can look at the j-value changes separately from the S.</p>
<p>j values meet the following pattern</p>
<p>Next[j]=0 j=1 hours</p>
<p>Next [j]=k exj-1 elements similar to prefixes and tailstips plus one allows for two selections of an element such as ababaa next[6]=3</p>
<p>Next [j] = 1 other </p>
<p>With these, we can get the code.</p>
<pre><code class="language-c">void get_next(String T,int *next){
    int i,j;
    i = 1;
    j = 0;
    next[1]=0;
    while(i&lt;T[0]){// 还是意味着长度
        if(j==0||T[i]==T[j]){
            i++;
            j++;
            next[i] = j;
        }
        else{
            j = next[j];
        }
    }
}
//这段代码是在生成我们需要的next数组来方便节约j的循环
int Index_KMP(String S,String T,int pos){
    int i = pos;
    int j = 1;
    int next[255];
    get_next(T,next);
    while(i&lt;=S[0]&amp;&amp;j&lt;=T[0]){
        if(S[i]==T[j]){
            j++;
            i++;
        }
        else{
//            i = i-j+2;
//            j = 1;
              j = next[j];   //这里是代码的核心变化
        }
    }
    if(j&gt;T[0]){
        return i-T[0];
    }
    else{
        return 0;
    }
}
//我们只是对原本的朴素匹配算法进行了轻微的调整 就对有很多部分匹配的效率进行了很好的优化 KMP算法还有更多的改进 我们在这里就不再叙述了 串这里引入算法部分只是希望我们有一个初步的理解 而非全部 这也是这里能做的全部了
</code></pre>
<h2>Tree</h2>
<p>The tree is a type of data structure that meets a large pair of data, and because of his characteristics, we learned that this structure can deal with many programming problems. </p>
<p><strong>The tree TREE is a limited collection of n nodes, n=0 is called an empty tree, and one of the non-empty trees, and only one of the specific nodes is the root root root of the tree, and the subnodes are a separate tree, called the subtree, which we have already talked about at the very beginning of the data structure.</strong></p>
<p>The roots of the tree are the only one, and he's not exactly the same as reality.</p>
<p>The number of subtrees that the node owns is called his degree, Degree.</p>
<p>The zero-degree node is called the leaf node, Leaf</p>
<p>The nodes of degrees are called non-end nodes or branch nodes, except for root nodes, which are also called internal nodes. </p>
<p>The tree size is the maximum number of nodes within the tree</p>
<p>Child, the root of the node tree is called the node child, and this node is, in turn, the node of his parents Parent, the node of his parents, the node of his parents, the node of his parents, the node of his brothers Sibling, and the node of his ancestors, from the node to the node, and the node of his children.</p>
<p>The idea of the tree exists, the first level, the first level, the second level, the next level, the constant analogy, the nodes of the same layer, the same level of the parents, the highest level of the tree, the depth of the tree, the Deepness of the Deep.</p>
<p>If the subtrees have a sequence from left to right that cannot be interchanged, they call him an orderly tree, and instead, an disorderly tree.</p>
<p>Forest Forest is a collection of trees that don't interact, and actually node subtrees form forests. </p>
<p>It's clear that trees have a very different structure and linear surface, and that they're much more complex. </p>
<p>The trees' ADT we're not going to repeat here, we can actually design it according to our needs. </p>
<h3>Tree Storage Structure</h3>
<p>The simple sequenced storage structure does not allow for the storage of trees, and we will design the structures and methods for storage of trees in combination with the storage methods in front of them.</p>
<h4>Parental expression</h4>
<p>In the parental representation, we store all the nodes in sequence in sequence, and each node's pointing field is designed to point to his parents, as follows:</p>
<pre><code class="language-c">typedef struct PTNode{
    ElemType data;
    int parent;
}PTNode;
typedef struct{
    PTNode nodes[MAX_TREE_SIZE];
    int r,n; //这个变量用来存储根节点的位置和结点的数目
}PTree;
</code></pre>
<p>We just need to be negative for the no-parent nodes, and in this design, it's difficult to find a child, and we need to go through the whole structure to do it.</p>
<pre><code class="language-c">typedef struct PTNode{
    ElemType data;
    int parent;
    int firstchild;
}PTNode;
</code></pre>
<p>It's obvious that this is not going to work on the brother situation, so we might as well add a brotherhood.</p>
<pre><code class="language-c">typedef struct PTNode{
    ElemType data;
    int parent;
    int firstchild;
    int rigthsib;
}PTNode;
</code></pre>
<p>The design of storage structures is very flexible, and whether it continues to grow depends on demand and not on other things.</p>
<h4>Child expression</h4>
<p>Because the tree nodes may have multiple sub-trees, so we're thinking about using multiple chains, and each has multiple points of reference pointing to the sub-tree nodes, and because the number of children is different, it's better to store his degree in the nodes to make it easier for us to achieve it, but it raises different issues about the nodes structure to deal with this difference. <strong>We chose to put the child nodes in each of the nodes, using a single chain table as a storage structure, which means a child node has a child watch.</strong> If the leaf nodes are empty, and finally ** the head pointers of these single-chain tables are stored in a linear table, which often uses sequential storage structures.** This is the child expression.** </p>
<pre><code class="language-c">typedef struct CTNode{
    int child;//这是用来标识这个孩子链表在我们的线性表的位置的
    struct CTNode *next;
}*ChildPtr;
//这是孩子链表的结点
typedef struct{
    ElemType data;
    ChildPtr firstchild;
}CTBox;
//这是一个普通的结点 他有数据域和孩子链表的指针域
typedef struct{
    CTBox nodes[MAX_TREE_SIZE]; //所有的结点顺序存储起来
    int r,n;  //存储根结点的位置和结点的数目
}CTree;
//这是在建立树
</code></pre>
<p>With this marking, it's easier to find children at nodes and brothers at nodes, but it's harder to find both parents, and we can certainly combine the benefits of the nodes and form the benefits of the nodes.<strong>Parental child</strong></p>
<pre><code class="language-c">typedef struct{
    ElemType data;
    ChildPtr firstchild;
    int parent;
}CTBox;
</code></pre>
<h4>The boy and brother say it's fair.</h4>
<p>We started with the nodes of parents and children, and we tried to express it from the brothers of both parents, and of course, it was impossible to form a tree structure. </p>
<pre><code class="language-c">typedef struct CSNode{
    ElemType data;
    struct CSNode *firstchild,*rigthsib;
}CSNode,*CSTree;
</code></pre>
<p>The group was not selected as we were at the very beginning because the pointer actually greatly increased the flexibility of the structure design, so that one tree that was made up of just a lack of access to both parents, and the rest was very comfortable, and the most beneficial of this expression was that it was actually a very good idea to have a good idea to have a good idea.<strong>Turned the original complex tree structure into a fork tree.</strong> And without the partial nodes, it gives us a very comfortable character.</p>
<p>It's the best way to express it, but actually, it's the designer who needs to design the data structure.</p>
<h3>Definition and nature of the diagonal tree</h3>
<p><strong>Bident tree Binary Tree</strong> He has a root and two non-intersected trees called the root and a two-knot tree. </p>
<ul>
<li>Two sub-trees at most fork tree at each node, one or zero.</li>
<li>The left tree and the right tree are in order.</li>
<li>Even if there's only one tree, it's a right tree.</li>
<li>He's in the same basic form.<ul>
<li>Empty fork tree</li>
<li>There's only one root.</li>
<li>The root is only the left tree.</li>
<li>The root is only the right subtree.</li>
<li>There's a left tree and a right tree at the root.</li>
</ul>
</li>
</ul>
<p>Here's some of the special fork trees.</p>
<ul>
<li>All the nodes are left or right, but the linear table is a special slash.</li>
<li>All the branches are left and right. All the leaves are on the same floor. Go, go, go!</li>
<li>The complete fork tree is numbered by the sequence of the layers of a fork tree with an n node. <ul>
<li>The leaves are only two floors down.</li>
<li>The lowermost leaves must be on the left side of the line.</li>
<li>The last layer of leaves must be in a continuous position on the right.</li>
<li>There is no right tree no more.</li>
<li>The same node fork tree, the smallest depth of the fork tree.</li>
</ul>
</li>
</ul>
<p><strong>Nature of the diagonal tree</strong></p>
<ul>
<li>The first layer has a maximum of 2 ^ (i-1) nodes, because it does not exceed the number of nodes full of fork trees.</li>
<li>The fork tree with a depth of k has a maximum of 2 k-1 nodes</li>
<li>Any two-knot tree if the terminal is 0 n and 2 degrees n2 n is 0 = n2 + 1</li>
<li>The depth of the full fork tree at n nodes is [log]<del>2</del>n]+1</li>
<li>For a full binary tree with n nodes, number the node in a stratification order for any node i<ul>
<li>At 1 p.m., he was root nodes without parents.&gt;Parents at 1st</li>
<li>If 2i&gt;Node i has no left child.</li>
<li>If 2i+1&gt;No, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no. No, no, no, no, no. No, no, no. No. I has no right, no, no, no, no, no, no, no, no, no, no.</li>
</ul>
</li>
</ul>
<h3>The store of the fork tree, the walk, the generation.</h3>
<p>As we have already said, the tree is a unique structure that is not easy to use in sequence storage, but the fork tree, because of its special nature, could also be considered as a more focused tool for us to study here rather than the sequence storage structure of the fork tree, which is, <strong>A table of the fork.</strong></p>
<pre><code class="language-c">typedef struct binode
{
    ElemType data;
    struct binode * lchild;
    struct binode * rchild;
}binode,*BiTree;
//实现存储是非常自然的 树指针自然的指向他的孩子 然后一直重复这个过程
</code></pre>
<p>The bident tree's history is certainly a very important part of the use of the didentary tree, and the core of it is that it has a sequence of access to all the nodes of the dident tree, which is guaranteed that all of the nodes are visited only once, and that one of four common cross-trip methods is explained later, and that they are essentially linearized, as the didentary tree is defined by the principle of regression, and we are going to go through it in a manner that understands the concept of retrogression as an important link here.</p>
<pre><code class="language-c">//前序遍历递归二叉树算法
void PreOrderTraverse(BiTree *T)
{
    if(T==NULL)
    return;
    printf(&quot;%c&quot;, T-&gt;data);          //显示结点数据，可以更改为其他对结点操作
    PreOrderTraverse(T-&gt;lchild);    //再先序遍历左子树
    PreOrderTraverse(T-&gt;rchild);    //最后先序遍历右子树
}
//中序遍历递归二叉树算法
void InOrderTraverse(BiTree *T)
{
    if(T==NULL)
    return;
    InOrderTraverse(T-&gt;lchild); //中序遍历左子树
    printf(&quot;%c&quot;, T-&gt;data);      //显示结点数据，可以更改为其他对结点操作
    InOrderTraverse(T-&gt;rchild); //最后中序遍历右子树
}

//后序遍历递归二叉树的算法
void PostOrderTraverse(BiTree *T)
{
    if(T==NULL)
    return;
    PostOrderTraverse(T-&gt;lchild);   //先后序遍历左子树
    PostOrderTraverse(T-&gt;rchild);   //再后续遍历右子树
    printf(&quot;%c&quot;, T-&gt;data);  //显示结点数据，可以更改为其他对结点操作
}
//差异其实就是打印位置的不同 本质上思路完全一样
</code></pre>
<p>The fact that the generation of the fork tree is a slight change in the way we think about it is, we give the order of the front lines, and we fill the lines in the tree.</p>
<pre><code class="language-c">//前序遍历递归法建立二叉树算法
void CreatBiTree(BiTree *T)
{
    char data;
    scanf(&quot;%c&quot;,&amp;data);
    if(data==&#39;#&#39;){
        T=NULL;
    }
    else
    {
        *T=(BiTree *)malloc(sizeof(BiTree));
        (*T)-&gt;data=data;
        CreatBiTree(&amp;(*T)-&gt;lchild);
        CreatBiTree(&amp;(*T)-&gt;rchild);
    }
}
</code></pre>
<pre><code class="language-c">//按层遍历递归二叉树算法 补充前面没提到的一个内容
void Layer_order(BiTree * TNode,BiTree ** F,BiTree ** R)  //二级指针
{

    *F=TNode;            //将当前节点放入队列首指针所指位置
    printf(&quot;%c&quot;,(*F)-&gt;data);
    if((*F)-&gt;lchild!=NULL)
    {
    R=R+1;
    *R=(*F)-&gt;lchild;    //节点的左儿子放入队尾
    }
    if((*F)-&gt;rchild!=NULL)
    {
    R=R+1;                //首指针向后移动一格
    *R=(*F)-&gt;rchild;      //节点的右儿子放入队尾
    }

    if(F!=R)

    {
     F=F+1;
     Layer_order(*F,F,R);//递归
    }
}
</code></pre>
<p>It's also a very good way to add the node to the parents to form a trident watch.</p>
<h3>Threads for the two fork tree, conversion of the two fork tree.</h3>
<p>The point of the clue didentary tree is to study the front and the follow-up of each node in a certain order, because the clueization itself is just saving space, so we don't have to describe it in detail here, and then we'll talk about it later.</p>
<p>If all the trees and forests are fork trees, it's going to be very easy to study, and we've been able to twig one tree with the child brother, in some abstract sense, the forkwatch is another expression of the fork tree, so we have to set certain rules that we can decimate even the tree.</p>
<p>For the ordinary tree, the boy brother would have been able to decipher it, and each node would have been directed to his eldest and right brother, and just reorder it into a fork tree.</p>
<p>For the forest, each tree is a brother, so we'll start with two fork trees, the first fork trees, and obviously we can attach the rest of the trees to the original fork tree as the right child.</p>
<p>The other way around is not complicated. </p>
<p>The way the trees go through the trees, the way we go through the fork trees, the way the trees go through the trees, the way the trees go through the trees, the way the brothers choose the children, the way the trees go through all the trees, the way they go through all the trees, the way they go through all the trees, the way they go through all the trees, the way they go through the trees, the way they go through the trees, the way they go through the trees, the way they go through the trees, the way they go through the trees, the way they go through the trees, the way they go through the trees, the way they go through the trees, the way they go, the way they go through the trees, the way they go, the way they go, the way they go through the trees, the way they go, the way they go, the way they go, the way they go, the way they go, the way they go, the way they go, the way they go, the way they go, the way they go, the way they go, the way they go, the way they go through all through all over all over the trees, the trees, the trees</p>
<h3>Hefmann Tree.</h3>
<p>Huffman's code is one of our most common compression codes, which is important for modern computer development, while ensuring that original files do not lose too much view accuracy.</p>
<p>Hefman code was invented by Mathematician Hefman, who used a special fork tree in the code, Hefman tree.</p>
<p>We need to introduce the concept of the length of the tree's path, which means the number of times that the two nodes need to be judged, and he has us getting the length of the tree's path (the length of the path of all nodes) and if we give each node a weight, we get the smallest of the right path's WLP WLP, which is called the Hefmann tree.  </p>
<p>Now, while we know about the Hefmann tree, how do we build him?</p>
<ol>
<li>All the nodes are each with their power values forming the root points.</li>
<li>In this pool, two small weights of nodes (trees) are selected to construct a fork tree, the weight of which is the weight of two subtrees and the right of the tree to add the tree to the front pool and remove the original tree.</li>
<li>Repeat the process of two until the tree turns into a tree.</li>
</ol>
<p>So far, the Hefmann tree has been constructed.</p>
<p>The Hefman tree is not our purpose, actually, the Hefman code is based on the Hefman tree, the code of zero or one in the right and left, which allows for smaller storage space for higher-frequency letters, which is data compression, and we don't talk about the problem with the Hefman code compression in practice, which is understood when it's used.</p>
<h2>Figure</h2>
<p>Figure, Graph, and the content of our research in the middle of the pomposium, is very similar, usually expressed as G (V, E) and the conglomerate of the edges, respectively. </p>
<p>In the study of the figure, we use the tops, Vertex, to describe nodes, just as we do the notion of the online tabulations, of the element, to sort out what you're doing in these concepts, which is an important part of our learning, and we clearly ask for -- the topsymmetry cannot be empty -- that there must be a topsymmetry, and the so-called sides are actually the topsymmetry.</p>
<ul>
<li><strong>No Side</strong> There's no difference of direction between the vertex. <strong>There's a side.</strong>Contrary to him, </li>
<li>Is there a pattern or not? </li>
<li>For those who have to go to the side we call his edge Edge Arc. </li>
<li>To facilitate the description, the un-supplied description is in parentheses, and the unsupplied description is in square brackets.</li>
<li>If there's no point to his own side and the same side does not repeat it, call him a simple picture.</li>
<li>In the no-go map, if all the vertexes are connected, he's called completely ungodly.<del>n</del>♪ Two sides ♪</li>
<li>If there's a graph, if all the vertex are connected in two opposite directions, he's called a complete graph 2 *C.<del>n</del>^2^</li>
<li>Because of the variable number of sides, there are concepts of thin and dense maps, and he has no quantitative criteria.</li>
<li>Some of the numbers that exist in the edges and arcs are called Right Weight, and the map with rights is called Network.</li>
<li>The concept of the submersible exists.</li>
<li>Introduction of the concept of a neighbourhood Adjacent and a dependency of an incidenent with two neighbours. Points </li>
<li>We can easily find that for the no-turn map, the number of sides is half the vertex and half the size of the vertex.</li>
<li>For those with an orientation map the number of edges equals the number of outs and equals the sum of the ins and outs.</li>
<li>From one vertebrae to another, we have the concept of a path path path path path path path path path, the length of which is the number of sides.</li>
<li>There's a Cycle concept of a return, also called a ring or a loop.</li>
<li>In the undirected map, if there's a path link at the top of the randomly chosen point, we call him Connected Graph, which is a very important concept to be studied later.</li>
<li>A large, unprompted satellite of connectivity is called the connectivity weight</li>
<li>In the flow map, if there are two-way connections, it's a strong connection map, with a strong connection sub-graph, a strong connection point.</li>
<li>And then we'll talk about the trees that generate the connectivity map, which is a tiny little connection sub-graph, and we'll have all the n-points in the map, but only the n-1 sides that make up a tree, and the n-1 sides are bound to be ringed, and the n-1 sides are not necessarily tree-generated.</li>
<li>If a directional map happens to have a vertex of zero and the rest of the vertex of one, it's a tree with a vertebrate, and it's the same thing to generate a forest, and it's better to understand the graph.</li>
</ul>
<h3>Figure ADT and storage structure</h3>
<p>The ADTs are more complicated than that, and we have the usual methods and are ready to add them to the actual needs of use.</p>
<p>It's very obvious that because the image is multiple-to-one and multi-trail features, it's not realistic or feasible to use sequenced storage structures, and multiple chains can achieve the structure, but it's a waste of a lot of pointer fields, so we're going to use some of the more interesting ways to store the chart structure, just like the tree structure.</p>
<h4>Adjacency Matrix</h4>
<p>The idea at the heart of this approach is to separate the top and the side, and the top points can be solved by using a 1-dimensional array, and the side can be addressed by using a 2-dimensional array, or by using it as a adjacent matrix.</p>
<p>If there's a side between the two nodes, the value of the adjacent matrix is one, and there's no side is zero, and if there's a direction map, it's just a change of indicator signs. </p>
<p>With the adjacent matrix, we can easily store a map.</p>
<p>In the front, we have the idea of a web, or we can change the symbol addition weight to achieve it.</p>
<pre><code class="language-c">typedef struct{
    VertexType vexs[MAXVEX];
    EdgeType arc[MAXVEX][MAXVEX];
    int numVertexes,numEdges;
}MGraph;
void CreateMGraph(MGraph *G){
    int i,j,k,w;
    scanf(&quot;%d%d&quot;,&amp;G-&gt;numVertexes,&amp;G-&gt;numEdges);
    for(i=0;i&lt;G-&gt;numVertexes;i++){
        scanf(&quot;%d&quot;,&amp;G-&gt;vexs[i])
    }
    for(i=0;i&lt;G-&gt;numVertexes;i++){
        for(j=0;j&lt;G-&gt;numVertexes;j++){
            G-&gt;arc[i][j] = 65535;//这是初始化的意思 权值不可能为这个值
        }
    }
    for(k=0;k&lt;G-&gt;numEdges;k++){
        scanf(&quot;%d%d%d&quot;,&amp;i,&amp;j,&amp;w);
        G-&gt;arc[i][j] = w;
        G-&gt;arc[j][i] = w;
    }//识别与创建部分
}
//很明显能看到 这个时间复杂度在n^2级别 并不算低
</code></pre>
<h4>Adjacency List</h4>
<p>It's clear that the matrix of the tie map is very expensive for storage space on the 2D matrix and that when the thin map is not working, we're thinking of using linear tables to save storage space, which is a similar idea to the first two ways we use in the tree sector.</p>
<p>The top one-dimensional arrays are used to store the chain tables, of course, and the top-point arrays also need to point the data elements to the first adjacent point, and we make all the adjacent points at each of the top points a linear table to save the storage space, and with a right value, the variable of field power values on each of the elements of the online table is the exact same principle, but simply adds the concept of a counternap table to the sky.</p>
<pre><code class="language-c">typedef struct EdgeNode{
    int adjvex;
    ElemType weigth;
    struct EdgeNode *next;
}EdgeNode;
//边结点 最后形成邻接表
typedef struct VertexNode{
    ElemType data;
    EdgeNode *firstdege;
}VertexNdoe,AdjList[MAXVEX];
// 顶点结点 只是用来做位置标识 其中要存储每个结点的邻接表
typedef struct{
    AdjList adjList;
    int numVertexes,numEdges;
}
//这就是最后的邻接表结构
//邻接表考虑出度 逆邻接表研究入度 他们的差异在有向图上面体现 本质山没什么不同
void CreateALGraph(GraphAdjList *G){
    ing i,j,k;
    EdgeNode *e;
    scanf(&quot;%d%d&quot;,&amp;G-&gt;numVertexes,&amp;G-&gt;numEdges);
    for(i=0;i&lt;G-&gt;numVertexes,i++){
        scanf(&quot;%d&quot;,&amp;G-&gt;adjList[i].data);
        G-&gt;adjList[i].firstedge=NULL;
    }
    for(k=0;k&lt;G-&gt;numEdges,k++){
        scanf(&quot;%d%d&quot;,&amp;i,&amp;j);
        e = (EdgeNode*)malloc(sizeof(EdgeNode));
        e-&gt;adjvex=j;
        e-&gt;next = G-&gt;adjList[i].firstedge;
        G-&gt;adjList[i].firstedge = e; //下面还要创建对等的
        e-&gt;adjvex=i;
        e-&gt;next = G-&gt;adjList[j].firstedge;
        G-&gt;adjList[j].firstedge = e;
        //这里的链表用的是头插法 前面已经介绍过了 逆邻接表和有向的和这个没有本质的区别
    }
}
</code></pre>
<h4>Cross-Clock</h4>
<p>The essence of the cross-Cross is that it combines the adjacent and the reverse edifice, so that we can study both the degree of exposure and the degree of input, and the structural changes are as follows:</p>
<pre><code class="language-c">typedef struct VertexNode{
    ElemType data;
    EdgeNode *firstin;
    EdgeNode *firstout;
}VertexNdoe,AdjList[MAXVEX];
</code></pre>
<p>His best use is to handle the flow map, because it's the only time he's gonna have to study the details. degrees</p>
<h4>Multiple Chains next to</h4>
<p>The multiple tables are an optimised structure for undirected maps, and his aim is to have two nodes in the table to describe one side to optimize. <strong>Understanding the Neighborhood Watch is a central step in our learning.</strong></p>
<h4>Sidesets</h4>
<p>The side array is a simpler way to store the information on the top point in one array, and the information on the side of the top point in one array, which is not suitable for common deletion, addition, search operation, just for all the sides to achieve a certain purpose, and we're going to mention some algorithms later on.</p>
<h3>TraversingGraph</h3>
<p>And obviously, we need to study this, but the picture is much more complicated than the tree, and we're doing it through the tree's bi-cords and using the boy's brother to decorate the original tree, but in the complex picture, we don't have the technique of marking the path through, and we can just keep it afloat.</p>
<p>Except for the repetition, the map has only one core element to go through. </p>
<p>There's no difference between having a graph and being untraceable.</p>
<h4>Depth Priority Search DFS</h4>
<p>The depth first search is essentially a regression process, and it's a little bit like a front-line search in front of us, where he keeps accessing until he finds unmarked elements, and then he returns, and we continue to look for the last layer back to the beginning, and we're going to complete the loop/search of this connection map, and if there's an unmarked node, it's definitely not in this map, and there's a need to re-engineer the DFS in the adjacent list, and the sequence is the simplest fixed order.</p>
<pre><code class="language-c">void DFS(GraphAdjList GL,int i){
    EdgeNode *p;
    visited[i] = TRUE;
    printf(&quot;%c&quot;,GL-&gt;adjList[i].data);
    p = GL-&gt;adjList[i].firstedge;
    while(p){
        if(!visited[p-&gt;adjvex]){
            DFS(GL,p-&gt;adjvex);
        }
        p = p-&gt;next;
    }
}
void DFSTraverse(GraphAdjList GL){
    int i;
    for(i=0;i&lt;GL-&gt;numVertexes;i++){
        visited[i] = FALSE;
    }
    for(i=0;i&lt;GL-&gt;numVertexes;i++){
        if(!visited[i]){
            DFS(GL,i);   //如果图是连通的 这个DFS只会执行一次
        }
    }
}
</code></pre>
<h4>Broad Search BFS</h4>
<p>The breadth priority is a layered approach, and the cross-border of a series of hands that goes at a distance of 1 and then the continuous marking of 2 is studied, and it's cyclically running, and all the elements that need to be out of the team are checked for a nearby point, which is the core of the BFS algorithm, and the time complexity of both algorithms is exactly the same, and the sequence of the front is important to understand whether the BFS is still the same or not, and both algorithms require knowledge.</p>
<pre><code class="language-c">void BFSTraverse(GraphAdjList GL){
    int i;
    EdgeNode *p;
    Queue Q;
    for(i=0;i&lt;GL-&gt;numVertexes;i++){
        visited[i] = FALSE;
    }
    InitQueue(&amp;Q);
    for(i=0;i&lt;GL-&gt;numVertexes;i++){
        if(!visited[i]){
		visited[i] = TRUE;
            printf(&quot;%c&quot;,GL-&gt;adjList[i].data);
            EnQueue(&amp;Q,i);
            while(!QueueEmpty(Q)){
                DeQueue(&amp;Q,&amp;i);
                p = GL-&gt;adjList[i].firstedge;
                while(p){
                    if(!visited[p-&gt;adjvex]){
                        visited[p-&gt;adjvex]=TURE;
                        printf(&quot;%c&quot;,GL-&gt;adjList[p-&gt;adjvex].data);
                        EnQueue(&amp;Q,p-&gt;adjvex);
                    }
                    p=p-&gt;next
                }
            }
        }
    }
}
</code></pre>
<h3>Minimal Generating Tree</h3>
<p>We mentioned that the tree that created the connectivity map was his tiny little connectivity sub-graph, which contained all the peaks, but only on the n-1 side, we called the smallest cost of constructing the network, obviously the right value, the tree that produced the Minimum Spanning Tree.</p>
<p>We're here to present two algorithms for the generation of the smallest tree, and it's not advisable to consider the least tree generation for negative edges and unconnected maps.</p>
<h4>Prim algorithm</h4>
<p>It's a little like a front-line dye process, choosing the smallest edge vertebrae to dye. </p>
<pre><code class="language-c">void MiniSpanTree_Prim(MGraph G){  //我们选择矩阵形式
    int min,i,j,k;
    int adjvex[MAXVEX];
    int lowcost[MAXVEX];
    lowcost[0]=0; //到集合的距离
    adjvex[0]=0;
    for(i=1;i&lt;G.numVertexes;i++){
        lowcost[i] = G.arc[0][i];
        adjvex[i] = 0;
    }
    for(i=1;i&lt;G.numVertexes,i++){
        min = INF;
        j=1;
        k=0;
        while(j&lt;G.numVertexes){
            if(lowcost[j]!=0&amp;&amp;lowcost[j]&lt;min){
                min = lowcost[j];
                k=j;
            }
            j++;
        }
        printf(&quot;(%d,%d)&quot;,adjvex[k],k);
        lowcost[k]=0;
        for(j=1;j&lt;G.numVertexes;j++){
            if(lowcost[j]!=0&amp;&amp;G.arc[k][j],lowcost[j]){
                lowcost[j]=Garc[k][j];
                adjvex[j]=k;
            }
        }
    }
}
</code></pre>
<h4>Kruskal algorithm</h4>
<p>And here we start with the edges, not the top points, and finding the smallest edges is the core of the area, and it's clear that this algorithm will have to judge whether the edges will be around the loop, and that's the most important thing about it.</p>
<p>To facilitate the back, it's not so hard to convert the matrix to a side array.</p>
<pre><code class="language-c">void MiniSpanTree_Kruskal(MGraph G){
    int i,n,m;
    Edge edges[MAXEDGE]; //按照权值排好顺序的代码我们省略了
    int parent[MAXVEX];
    for(i=0;i&lt;G,numEdges;i++){
        parent[i]=0;
    }
    for(i=0;i&lt;G.numEdges,i++){
        n=Find(parent,edges[i].begin);
        m=Find(parent,edges[i].end);
        if(n==0||m==0){
            parent[edges[i].begin]=1;
            parent[edges[i].end)]=1;
            printf(&quot;(%d %d) %d&quot;,edges[i].begin,edges[i].end,edges[i].weight)
        }
    }
}
int Find(int *parent,int f){
    while(parent[f]!=0){
        return 1;
    }
    return 0;
}
</code></pre>
<p>Two algorithms are obvious, one for a relatively small number of scenarios, and the right choice is our core, understanding the algorithm and writing about what we're doing is the point.</p>
<h3>Shortest Path Problem</h3>
<h4>Dijkstra algorithm</h4>
<p>This is an algorithm that selects the shortest path by increasing the length of the path, an improvement from the BFS algorithm, and a continuous extended search for recently found nodes.</p>
<pre><code class="language-c">int Pathmatirx[MAXVEX];  //前驱顶点的下标 实际上就是路径数组 它存储的是路怎么走
int ShortPathTable[MAXVEX]; //最短路径的存储
void ShortestPath_Dijkstra(MGraph G,int v0,Pathmatirx *P,ShortPathTable *D){
    int v,k,w,min;
    int final[MAXVEX]; //存储这个顶点有没有找到最短路径的状态 1就是找到了 0是没找到
    for(v=0;v&lt;G.numVertexes;v++){
        final[v]=0;
        (*D)[v] = G.matirx[v0][v];
        (*P)[v] = 0;
    }
    (*D)[v0] = 0;
    final[v0] = 1; //第一个顶点的路径已经确定 到本身
    for(v=1;v&lt;G.numVertexes;v++){
        min = INF;
        for(w=0;w&lt;G.numVertexes;w++){
            if(!final[w]&amp;&amp;(*D)[w]&lt;min){
                k = w;
                min = (*D)[w];
            }
        }
        final[k] = 1;
        for(w=0;w&lt;G.numVertexes;w++){
            if(!final[w]&amp;&amp;(min+G.matrix[k][w])&lt;(*D)[w]){
                (*D)[w] = min+G.matrix[k][w];
                (*P)[w] = k;
            }
        }
    }
}
</code></pre>
<h4>Floyd algorithm</h4>
<p>This algorithm is the shortest path between all points and all points at the same time, so that all points can try to be relayed to see if they can optimize the algorithm.</p>
<pre><code class="language-c">int Pathmatirx[MAXVEX][MAXVEX];
int ShortPathTable[MAXVEX][MAXVEX];
void ShortestPath_Floyd(MGraph G,Pathmatrix *P;ShortPathTable *D){
    int v,k,m;
    for(v=0;v&lt;G.numVertexes;v++){
        for(w=0;w&lt;G.numVertexes;w++){
            *D[v][w] = G.matrix[v][w];
            *P[v][w] = w;
        }
    }
    //初始化的过程
    for(k=0;k&lt;G.numVertexes;k++){
        for(v=0;v&lt;G.numVertexes;v++){
            for(w=0;w&lt;G.numVertexes;w++){
                if((*D)[v][w]&gt;(*D)[v][k]+(*D)[k][w]){
                    (*D)[v][w]=(*D)[v][k]+(*D)[k][w];
                    (*P)[v][w]=(*P)[v][k]
                }
            }
        }
    }
    //这个直接对原始的修正是非常巧妙的 很简洁的代码实现了很复杂的功能
}
</code></pre>
<h3>Scaled up</h3>
<p>We've finished with two revolving applications, and now we've got to think about the unringing applications -- the scaling-up of the sequence -- which is essentially a no-go-program, obviously the event is not going to happen, and the conditions are not going to work until we finish the follow-up, so we've introduced the AOV network -- the Activity On Network -- and we've introduced the idea of a poking sequence in the process of dealing with the AOV network, and if there's a directional path, he's just going to expand the sequence. </p>
<p>So the sort of thing that goes into the scale is a process that has a tectonic extension sequence, and if all the vertexes are exported, it proves there's no loop back, or he's not AOV, and it's obviously the way that the sequence is done.</p>
<p>The basic idea of scaling up the sorting selects the point output with an input of zero, removes this vertex and arc with his end in it repeats the process </p>
<p>To facilitate the removal of the adjacent forms, we need to create a adjacent form that we can use here, and he needs to focus on the input, which is not important.</p>
<pre><code class="language-c">/* 拓扑排序，若GL无回路，则输出拓扑排序序列并返回OK，若有回路返回ERROR */
Status TopplogicalSort(GraphAdjList GL){
	EdgeNode *e;
	int i, k, gettop;
	int top = 0;	//用于栈指针下标
	int count = 0;	//用于统计输出顶点的个数
	int *stack;	//建栈存储入度为0的顶点
	stack = (int *)malloc(GL-&gt;numVertexes * sizeof(int));
	for(i=0;i&lt;GL-&gt;numVertexes;i++)
		if(GL-&gt;adjList[i].in == 0)
			stack[++top] = i;	//将入度为0的顶点入栈
	while(top != 0){
		gettop = stack[top--];	//出栈
		printf(&quot;%d-&gt;&quot;,GL-&gt;adjList[gettop].data);	//打印此顶点
		count++;		//统计输出顶点数
		for(e = GL-&gt;adjList[gettop].firstedge;e;e = e-&gt;next){	//对此顶点弧表遍历
			k = e-&gt;adjvex;
			if(!(--GL-&gt;adjList[k].in))	//将k号顶点邻接点的入度减1
				stack[++top] = k;	//若为0则入栈，以便下次循环输出
		}
	}
	if(count &lt;GL-&gt;numVertexes)	//如果count小于顶点数，说明存在环
		return ERROR;
	else
		return OK;
}
</code></pre>
<h3>Key Path</h3>
<p>This is a little bit like a sort of sort of sort of sort of sort of a hype, which is a kind of unringing application, and all of them need to be doing things in a similar way, but the key path question needs time, if the AOC network takes time-right, and he becomes AOE, and we're not doing the order of work, but we're looking at the time, the longest path from source to sink, which we call key activities, and the activity above is called critical activities, and to improve overall efficiency, it's going to be from key activities, and obviously the key path algorithm and the top-up sequence are very similar to what we're doing. </p>
<p>Several key parameters</p>
<p>The earliest time of the event is tv: the earliest time of the vertex vk;
The latest event time is at ltv: the last time at the top vk, the last time at each vertex that the event needs to start at the latest, and the time that will be exceeded will delay the entire period;
The earliest start time of the activity is et: the earliest time of the arc;
The last time the activity starts is the last time the arc aak occurs, the last time the work starts without delay.
3 and 4 can be determined by 1 and 2 and then whether or not ak is a key activity depending on whether or not ete[k] is equal to the lte[k]</p>
<pre><code class="language-c">int *etv, *ltv;	//事件最早发生时间和最迟发生时间数组
int *stack2;	//用于存储拓扑序列的栈
int top2;	//用于stack2的指针
//改进的拓扑排序代码 用于研究关键路径问题
Status TopologicalSort(GraphAdjList GL){
	EdgeNode *e;
	int i, k, gettop;
	int top = 0;	//用于栈指针下标
	int count = 0;	//用于同级输出顶点的个数
	int *stack;	//建栈将入度为0的顶点入栈
	stack = (int*)malloc(GL-&gt;numVertexes * sizeof(int));
	for(i=0;i&lt;GL-&gt;numVertexes;i++)
		if(0 == GL-&gt;adjList[i].in)
			stack[++top] = i;
	top2 = 0;	//初始化为0
	etv = (int*)malloc(GL-&gt;numVertexes*sizeof(int));	//事件最早发生时间
	for(i=0;i&lt;GL-&gt;numVertexes;i++)
		etv[i] = 0;	//初始化为0
	stack2 = (int*)malloc(GL-&gt;numVertexes*sizeof(int));	//初始化
	while(top!=0){
		gettop=stack[top--];
		count++;
		stack2[++top2] = gettop;	//将弹出的顶点序号压入拓扑序列的栈
		for(e=GL-&gt;adjList[gettop].firstedge;e;e=e-&gt;next){
			k = e-&gt;adjvex;
			if(!(--GL-&gt;adjList[k].in))
				stack[++top] = k;
			if((etv[gettop]+e-&gt;weight &gt; etv[k])	//求各顶点事件最早发生时间值
				etv[k] = etv[gettop] + e-&gt;weight;	//前一个结点得权值加上当前边的权值，如果大于当前结点已经得到的权值，那么替换，得到当前结点最早发生时间值
		}
	}
	if(count &lt; GL-&gt;numVertexes)
		return ERROR;
	else
		return OK;
// 15-19 23 28 29行发生了变化 理解变化的意义

/* 求关键路径，GL为有向图，输出GL的各项关键活动 */
void CriticalPath(GraphAdjList GL){
	EdgeNode *e;
	int i, gettop, k, j;
	int ete, lte;	//声明活动最早发生时间和最迟发生时间量
	TopologicalSort(GL);	//求拓扑序列，计算数组etv和stack2的值
	ltv = (int*)malloc(GL-&gt;numVertexes*sizeof(int));	//事件最晚发生时间
	for(i=0;i&lt;GL-&gt;numVertexes;i++)
		ltv[i] = etv[GL-&gt;numVertexes-1];	//初始化ltv，初始化为最后那个结点的最早开始时间
	while(top2 != 0){	//计算ltv
		gettop = stack2[top2--];	//将拓扑序列出栈，后进先出
		for(e=GL-&gt;adjList[gettop].firstedge;e;e=e-&gt;next){	//求各顶点事件的最迟发生时间ltv值
			k = e-&gt;adjvex;
			if(ltv[k]-e-&gt;weight &lt; ltv[gettop])	//求各顶点事件最晚发生时间ltv，其中，gettop点为k结点的前一个结点.ltv[gettop]为已知的该结点最晚发生时间
				ltv[gettop] = ltv[k] - e-&gt;weight;
		}
	}
	for(j=0;j&lt;GL-&gt;numVertexes;j++){	//求ete，lte和关键活动
		for(e=GL-&gt;adjList[j].firstedge;e;e=e-&gt;next){
			k=e-&gt;adjvex;	//拿到邻接点下标
			ete = etv[j];	//活动最早发生时间
			lte = ltv[k] - e-&gt;weight;	//活动最迟发生时间
			if(ete == lte)	//两者相等即在关键路径上
				printf(&quot;&lt;v%d,v%d&gt; length:%d,&quot;,GL-&gt;adjList[j].data, GL-&gt;adjList[k].data, e-&gt;weight);
		}
	}
}
//如果有多个关键路径 影响一条是无效的 徐涛对多条关键路径下手
</code></pre>
