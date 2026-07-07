---
title: "数据结构导论：线性表、树、图与查找排序"
title_en: "Introduction to Data Structures: Linear Lists, Trees, Graphs, and Search"
date: 2025-05-12 22:24:30 +0800
categories: ["Programming", "Computer Science Fundamentals"]
tags: ["Data Structures", "Algorithms"]
author: Hyacehila
excerpt: "整理数据结构基本概念、线性表、栈、队列、树、图、查找和排序等程序设计基础。"
excerpt_en: "Introduces data structures through core concepts, linear lists, stacks, queues, trees, graphs, search, sorting, and programming fundamentals."
mathjax: false
hidden: true
permalink: '/blog/2025/05/12/data-structures-introduction/'
---
在这部分的学习中，我们会学到数据结构与算法的基础知识，当然更复杂的算法要在以后的别的教材中才能学到，不过这是一门程序设计的重要基础课，而且难度并不低

## 数据结构绪论

​	在早期的计算机发展过程中，数值计算是它需要解决的问题，所以处理问题的时候往往采用**抽象模型，设计算法，编写程序**的流程；但是非数值问题逐渐浮现，我们的思路在这里产生的转变；**数据结构是研究非数值计算的程序设计计算机中操作对象和关系的学科** 在这之后，处理问题的思路转换为了**选择结构类型 设计算法**的过程 此时 **程序设计=数据结构+算法**

### 基本概念和术语

#### 数据

数据是描述客观事物的符号，是计算机中可以操作的对象，是能被计算机识别并输入给计算机处理的符号集合 无论是数值数据 还是非数值编码形成的数据 都是我们这里要研究的数据

数据元素是组成数据的有一定意义的基本单位 往往在计算机中我们往往把它视为一个整体处理（类比一个对象）

数据项是数据不可分割的最小单位 是数据元素的组成（类比对象的属性） 请注意 数据结构这门学科很少的研究数据项

数据对象是性质相同的数据元素的集合（比如把同一类建立的多个对象进行归类）

数据结构是存在联系的数据元素的集合 正是因为数据元素不是孤立存在的 我们在设计程序的时候才要考虑他们的联系

#### 逻辑结构和物理结构

**逻辑结构表示数据对象中数据元素之间的相互关系** 这是我们要非常重视的一个问题

* 集合结构：所有元素平等
* 线性结构：数据元素之间存在一对一的关系 A对应B B对应C ...这种 
* 树形结构：一对多的层次关系
* 图形结构：多对多的关系 

我们一般使用示意图来表示他们的联系 要对图形基本有数 每一个节点都代表一个数据元素 连线如果没有箭头则表示双向 元素的前驱和后继都是重要的 

**物理结构是计算机的存储结构** 

* 顺序结构 C语言中的数组就是这样的
* 链式结构 C语言中的链表就是这样的 

逻辑结构面向问题 物理结构面向计算机 得到两者的桥梁是重要的

#### 抽象数据类型

**数据类型是指一组性质相同的值的集合和定义在这个集合上的一些操作的总称**

对数据类型的划分是为了根据需要决定数据占用的空间 是对计算机资源节约的一个手段 

* 原子类型：不可再分的基本类型 整型 实型 字符型等
* 结构类型：多个原子类型组合而成的 比如数组  

抽象数据类型是对已有数据类型特点的抽象 旨在研究事物具有的普遍性本质 我们一方面使用一些约定俗成的数据类型，一方面根据自己的使用需求研究属于自己得数据类型，比如结构 字典 类

ADT（抽象数据类型）应该包括 数据类型的名字 数据元素和逻辑关系的定义 对操作和操作结果的描述 用代码实现ADT是我们后面经常要处理的事情

## 算法

算法和数据结构是不能分开的两个要素，但是算法太复杂了，我们更多的会在更深入的可能中学习，这里我们只需要一些比较基础的算法知识保证数据结构能继续讲下去

### 算法定义与特性

算法是解决特定问题步骤的描述，在计算机中就是一些指令序列；很明显的，这世界上不存在通用的算法，这就是他复杂的缘故

#### 输入输出

这是非常好理解的 算法可以没有输入但是基本都有输出 没有输出算法毫无意义

#### 有穷性

算法不应该出现无限循环无法终止

#### 确定性

算法是精确地，每一步的含义都是固定的

#### 可行性

如果你的算法过于复杂以至于完全无法实现 这个算法就没什么意义了

### 设计要求与效率的度量

算法不是唯一的，所以我们会对这些不同的的算法有一些要求和衡量算法的好坏

#### 正确性

输入 输出 加工处理不得有歧义 应当能正确的处理问题 得到问题的正确答案 层次如下 一般我们无法保证一个算法完全正确 所以实现第三步一般就够了

* 无语法错误
* 合法输入返回正确的结果
* 非法输入给出适当提示
* 对于故意的测试数据集也能正确的返回

#### 可读性

如果其他人无法理解你的算法 那么他很难继续进步 所有保证算法的可读性很重要

#### 健壮性

对非法数据的合适处理 不要产生异常和莫名其妙的结果

#### 时间效率高和存储量低

就是时空复杂度的体现

#### 事后统计方法

通过编制测试数据集 运行程序的手段来衡量算法的时间复杂度 但是这需要耗费精力进行测试 容易被设备状况影响 所以很少采用他来进行衡量算法

#### 事前估算方法

首先我们需要明白问题输入的规模 他一般会用n来表示 对于两个不同设计目的一样的算法用同样的问题输入规模 衡量基本操作次数 把他们构成函数 就是我们想要研究的时间复杂度 随着规模不断增大，时间复杂度不同的算法消耗的计算资源数量不断扩大，最后形成质的差距

### 函数的渐进增长与时间复杂度

如何根据算法的时间复杂度函数直接评价算法的好坏？ 我们给出一些辅助的结论

**函数的渐进增长** 如果在n>N后 一个函数一直大于另一个 就称为他的增长渐进比另一个快 

此时我们可以得到一些辅助结论 他们都是数学分析的基本推出结论

* 加法常数可以忽略
* 最高次项的乘数可以忽略
* 非最高次项可以忽略
* 最高次项的阶数非常重要

根据前面结论的简化 事实上我们只用关注阶数就好了 O（1）O（n）O（n^2）就是我们需要关注的时间复杂度 一般称为O阶 对于O阶的研究 一个数学人应该很容易想到 注意一下 不存在O(12)这种东西 不过O（lgn）O（lnn）O（m*n）O（nlgn）O(2^n) 都是存在的 他们之间的大小关系不就是无穷大的比较 应该对于数学人也不是问题

### 最坏与平均情况

平均和最坏都是有意义的 但是最好一般是无意义的 

最坏是一种保证 他保证不会有更坏的情况发生了 所以我们往往研究他作为衡量标准

平均是所有情况最优意义的 但是我们难以用技术手段分析 只能上世纪数据测试

### 空间复杂度

对于空间复杂度我们往往要求不高 实际上在算法设计中我们经常采用空间换时间的手段 毕竟用户不一定会因为多占用一点存储多说什么 但是会对自己遇到的卡顿给出评价 实现完成一些常用的内容的计算 在需要的时候直接调用 是很好用的手段

降低代码的时间复杂度 让代码能在更古老的设备上更快速的运行 这是一个程序员在设计底层程序应该考虑到的 不过对于一个开发者看起来不是很重要

空间复杂度O(1) 意味着所有操作不会额外消耗数量级的内存 更高的空间复杂度同理

## 线性表

线性表是最简单并且最常用的数据结构之一，他有头 有尾 有顺序；

线性表：**零个或者多个数据元素的有限序列**

序列 意思是元素之间具有顺序 有限则是计算机的特性 只有数学才会处理无限的序列 理解线性表这个概念并不困难 比较复杂的线性表中每个数据元素可能具有多个数据项 这当然是符合数据结构这门学科的 深入的理解这个概念并且记忆到心中是目前的要求

### 线性表的抽象数据类型

* 创建
* 初始化
* 重置为空
* 借助位序随时进行访问
* 查找
* 访问总长度
* 插入和删除数据

目前我们认为这些操作是比较重要的 当然我们可以随时增加新的操作 不过我们不妨把这些认为是线性表的基本的操作

至于线性表的属性 其实只要理解线性表的概念就可以了 最核心的就是有限序列这一条

### 顺序存储结构

我们首先想到使用顺序存储结构来存储这个线性表 也就是找一块连续的内存来存放 由于线性表的特性，我们借助一维的数组就是一个相当不错的办法 那么这将是我们在数据结构这门课里面写下的第一行代码

```c
#define MAXSIZE 20
typedef int ElemType;
typedef struct
{
    ElemType data[MAXSIZE];
    int length;
}Sqlist;
//存储的类型是自由变化的 是结构也无所谓 这个我们这里强调一次 后面就不提的 很明显的我们选择使用这个结构作为存储数据的形式
//起始位置 数组data 最大容量 长度 均在我们的线性表中被体现的出来
```

数组的长度和线性表的长度是不一样的 数组在不引入VLA的时候就是固定的 引入之后是随代码运行固定的 线性表长度和存储的元素数量有关系 是动态的

关于地址关系我们就不在这里赘述了 C语言基础已经讲的很明白了

```c
#define OK 1
#define ERROR 0 
#define TRUE 1
#define FALSE 0
#define int Status 
Status GetElem (Sqlist L,int i,ElemType *e)
{
    if(L.length==0 || i<1 || i>L.length){
        return ERROR;
    }
    *e=L.data[i-1];
    return OK;
}
//建立访问线性表中元素的函数 由于不需要修改所以选择值传递
Status ListInsert(Sqlist *L,int i,ElemType *e)
{
    int k;
    if(L->length==MAXSIZE){
        return ERROR;
    }
    if(i<1||i>L->length+1){
        return ERROR;
    }
    if(i<=L->length){
        for(k=L-length-1;k>=i-1;k--){
            L->data[k+1]=L->data[k];
        }
    }
    L->data[i-1]=e;
    L->length++;
    return OK;
}
//检验插入位置后 插入元素到原本的线性表里面 这个程序的设计拒绝我们把元素跳跃的插入 否则线性表的前后元素就会出问题
Status ListDel(Sqlist *L,int i,ElemType *e)
{
    int k;
    if(L->length==0){
        return ERROR;
    }
    if(i<1||i>L->length){
        return ERROR;
    }
    *e = L->data[i-1] //返回一下被删掉的元素
    if(i<=L->length){
        for(k=i;k<L->length;k++){
            L->data[k-1]=L->data[k];
        }
    }
    L->length--;
    return OK;
}
//删除我们不需要的元素并且给出了被删除的元素
//如果要研究这几个函数的时间复杂度的话 查询算法复杂度1 后面的平均算法复杂度是n（研究平均才有意义）
//设计其他的函数其实并不是一个复杂的事情 在这里我们就留作练习了 考虑的细致周到一点就好
//整表的创建就是建立一个结构初始化后并且填入我们的初始数据 清为空就是控制长度为0 你不需要告诉我们存储了什么 只要知道哪些数据是无效的就好了 删除就是释放所有占用的内存 都不困难
```

### 链式存储结构

链式存储结构的出现对于数据结构是非常重要的 后面的很多逻辑结构用顺序很难或者根本无法进行存储 链式存储结构的核心就是随意的使用没有被占用的内存 对于内存的节约也是非常明显的 链式存储结构的重要点就在于每一个数据项都会存储下一个数据项的地址 这是一个为了数据结构形成而增加的数据项

我们把指针域和数据域的整体合并称为一个节点 Node 很明显这里我们只会用到指针域为一的情况 他也叫单链表 

单链表的第一个元素的位置被称为头指针 最后一个一个元素指针指向	NULL   有时候我们会建立**头节点存储头指针的信息和链表总体的一些信息** 或者干脆头结点就是一个指针指向下一个元素**（头指针）** 如果线性表为空 则头节点就指向NULL

```c
typedef struct Node
{
    ElemType data;
    struct Node *next;
}Node;//结点建立

typedef struct Node *LinkList; //链表建立 这一步完全可以浓缩到上一步 在栈里面我们就能看到这个写法 这里其实我们没有准备头节点来存储一些信息 
// 这就是最基本的节点的构成 由指针域和数据域 至于数据域究竟是什么样子我们这里不重视
Status GetElem(LinkList L,int i,ElemType *e){
    int j;
    LinkList p;
    p = L->next;
    j=1;
    while(p&&j<i){
        p = p->next;
        ++j;
    }
    if(!p||j>1){
        return ERROR;
    }
    *e = p->data;
    return OK;
}
//顺序访问我们的单链表 知道找到元素 很明显在这个查找时间复杂度是n优势不明显 由于单链表没有控制表长 所以我们不方便使用for循环
Status ListInsert(LinkList *L,int i,ElemType e)
{
	int j;
    LinkList p,s;
    p = *L;
    j=1;
    while(p&&j<i){
       p = p->next;
       ++j; 
    }
    if(!p||j>1){
        return ERROR;
    }
    s = (LinkList)malloc(sizeof(Node));
    s->data = e;
    s->next = p->next;
    p->next = s;
    return OK;
}
//这是插入单链表的方法 其本质并不复杂 采用malloc申请内存是为了处理C语言的变量生存期问题
Status ListDel(LinkList *L,int i,ElemType *e)
{
	int j;
    LinkList p,s;
    p = *L;
    j=1;
    while(p->next&&j<i){
       p = p->next;
       ++j; 
    }
    if(!(p->next)||j>1){
        return ERROR;
    }
	q = p->next;
    p->next = q->next;
    *e = q->data;
    free(q);
    return OK;
}
//这是删除 我们用free释放空间 请在数据结构里面避免指针运算 因为这是C语言特性 数据不只在C语言使用
//明显的我们能看到 链式存储结构在删除和插入有时间复杂度的优势
```

**整表的创建和删除** 在链式结构中我们应该单独考虑删除和创建的方法

```c
//单链表的整表创建就是循环建立元素节点并且依次连接形成的
void CreateList(LinkList *L,int n){
    LinkList p;
    int i;
    *L = (LinkList)malloc(sizeof(Node));
    (*L)->next = NULL;
    for(i=0;i<100;i++){
        p =  (LinkList)malloc(sizeof(Node));
        p ->data = 1;
        p->next = (*L)->next;
        (*L)->next = p;
    }
}
//这是头插的结构 新的节点一直在头指针L和第一个元素之间 不断补充新的第一个元素 
void CreateList(LinkList *L,int n){
    LinkList p;
    int i;
    *L = (LinkList)malloc(sizeof(Node));
    r = *L;  //结尾节点设置 
    for(i=0;i<100;i++){
        p =  (LinkList)malloc(sizeof(Node));
        p ->data = 1;
        r->next = p;
        r=p;
    }
    r->next = NULL;
}
//这就是尾插的方法 新的节点一直在旧的节点的尾部 
//如果我们已经完成了整表的创建想要添加新的元素直接用Insert的方法就可以 这里只是从零创建需要的
Status ClearList(LinkList *L){
    LinkList p,q;
    p = (*L)->next; //第一个结点
    while(p){
        q=p->next;
        free(p);
        p=q;
    }
    (*L)->next = NULL; //头结点的释放
    return OK;
}
//这是对整个单链表的释放 我们挨个存储下一个节点并且释放上一个节点 最后让头结点指针域归NULL
```

对单链表的讲解暂且就到这里了 我们很容易发现 链式存储结构的巨大优点就是便于插入和删除 顺序存储结构更适合访问并且长度有限 后面我们会见到一个有趣的链表结构 都和单链表脱不开干系 事实上后面许多结构复杂的数据结构都是最简单的结构逐渐增加复杂度实现的 理解单链表 你对数据结构基本就有了基础的认识

#### 静态链表

静态链表是一个非常特殊的产物 在C语言引入指针这一重要的内存操作手法后 后续的高级语言诸如Python和Java都启用的对象引用的制度（也就是面向对象语言）这样他们也能够以一个比较容易的方式实现数据结构 但是更早期的语言并没有这些手法 所以有的程序员选择用数组来一定程度上实现单链表的功能 也就是静态链表 不过他只是一个权宜之计 失去了链式存储结构的重要优点——灵活使用内存

#### 循环链表

单链表最大的问题就是只能从头开始访问 但是循环链表就可以一定程度解决这个问题 

循环链表的形成非常简单 只是把终端结点的指针域从空指针改为指向头结点 这样整个链表形成了一个环 他是单链表的一种特例 

在循环链表中 为了方便更多的代码设计 头结点基本是一定存在的 已经形成了潜规则 

循环链表和单链表最大的不同就是判断循环的条件不同

偶尔的我们会引入尾指针来方便对元素的访问 不过这并不常用 直接沿用单链表的处理方法就可以很好的处理循环链表了

#### 双向链表

非常自然的，我们会引入既能查看后继元素 也能查看前驱元素的链表 这样我们就可以克服单向性的缺点 也就是指针域被扩大了

```c
typedef struct DulNode
{
    ElemType data;
    struct DulNode *prior;
    struct DulNode *next;
}DulNode,*DulLinkList;
//写出一个双向链表的结点并不困难
//其他的函数其实也非常好写 只是更加繁琐了 细致细心 对齐各个指针就可以
```

## 栈与队列

这里我们其实是在研究两种特殊的线性表

**栈（stack）是限定在表尾进行插入和删除操作的线性表**

我们把允许进行插入和删除操作的一段称为栈顶top 另一端是栈底bottom 不含任何元素的称为空栈 实际上栈是一个后进先出的结构 LAST IN FIRST OUT 简称LIFO结构 我们要理解栈的最核心就是 栈是一类特殊的线性表 他从表尾或者说栈顶进行操作

很明显 我们认为栈的ADT和线性表应该没多少不同 只是del和insert函数应该改为push和pop来契合栈的特殊性

### 顺序存储结构

```c
typedef struct{
    ElemType data[MAXSIZE];
    int top; //记录栈顶的位置 top=0 代表元素只有一个 top为-1意味着栈是空的
}Sqstack;
//建立顺序存储结构的栈
Status Push(SqStack *S,ElemType e){
    if(S->top == MAXSIZE-1){
        return ERROR;//满栈
    }
    S->top++;
    S->data[S->top]=e;
    return OK;
}
//这是压栈的函数
Status Pop(SaStack *S,ElemType *e){
    if(S->top == -1){
        return ERROR;//空栈
    }
    *e = S->data[S->top];
    S->top--;
}
//这是出栈的函数 实际上只要借助top的位置就知道存到哪里了 不需要单独的初始化操作 这是我们现在的代码书写比较特殊的一点
```

其他的函数我们在这里就不写了 都不是很困难 理解方法这里就足够了

#### 两个栈共享存储空间

有时候我们会遇到数据类型一样的两个数据结构 他们有着此消彼长的关系 此时为了节约存储空间我们就会用到这个手法。这仅仅是一个存储上的技巧，不代表我们一定要用它 只用几个函数稍微的实例一下而不详细叙述

```c
typedef sturct{
    ElemType data[MAXSIZE];
    int top1;
    int top2;
}SqDoubleStack;//增加指针的数量
Status Push(SqDoubleStack *S,ElemType e,int StackNumber){
    if(S->top1+1==S->top2){
        return ERROR;
    }
    if(StackNumber==1){
        S->data[++S->top1]=e;
    }
    else if(StackNumber==2){
        S->data[--S->top2]=e;
    }
    return OK;
}
//基本上就是这样的思路 额外增加一个选择你进什么栈的选择 出栈原理也是完全一样的 实际书写代码的时候完全可以把修正栈顶和赋值语句分开分布
```

### 链式存储结构

由于单链表结构有指针的存在 栈的结构也需要栈顶来满足我们的进出需求 很容易理解把他们两者合成一个位置 此时我们发现头节点确实没啥用了 判断空完全可以用栈顶的指针是不是NULL来确定

```c
typedef struct StackNode{
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
    s->data = e;
    s->next = S->top;
    S->top = s;
    S->count++;
    return OK;
}
//实际上原理非常的简单 我们最重要的是理解这里指针的情况就可以解决
Status Pop(LinkStack *S,ElemType *e){
	*e = S->top->data;
    if(StackEmpty(*S)){
        return ERROR;
    }
    LinkStackPtr s = S->top
    S->top = S->top->next;
    free(s);
    S->count--;
    return OK;
}
//弹出栈的函数其实也很好写 其实最难的是C语言指针比较复杂的规则
```

栈已经被很多的高级语言进行了封装  这个原理其实看看就好

### 栈的应用

函数的递归是一个比较常用的程序设计思路 我们容易观察到 当函数一层一层的递归产生结果后 他们会回到上一次进行运算 这就是栈的思路 最后被压入栈的数据需要最先的出栈参与运算 当然目前高级语言会自动管理 我们只用负责递归就可以了

**队列（queue）也是一种特殊的线性表，他允许在一端进行插入操作，另一端进行删除操作** 意思是队列是First In First Out 的数据结构 允许插入的称为队头 允许删除的称为队尾 其实队列很符合我们生活中的习惯 所以应用非常广泛 比如键盘输入数据给系统 都是先输入的先进入系统 

### 顺序存储与循环队列

设计一个顺序存储结构的队列应该不是任何问题 想想 我们只需要在入队的时候在数组末尾加一个新元素 在出队的时候把第一个元素送走 让后面的补上  但是这样出队列的时间复杂度是不是太高了？ 

我们能不能用一点比较特殊的手法 比如让队头的位置进行一些改变 这样可以有效的降低时间复杂度 当然需要队头队尾两个指针

但是这样肯定会带来新问题 队头和队尾指针重合这个数组是空的 但是满数组不好判定了 尾指针在末尾的时候可不一定满数组 前面可能还有空间 这就是假溢出 怎么处理呢？

循环队列给出的方法就是 如果检测到尾指针rear到了结尾 我们就数组头开始  继续加入数据 直到rear追上的front 此时又引出了一个问题 重合同时代表满和空 处理的思路也很简单 如果发现rear即将追上front 就留下一个空位 直接报告满队列 这个空间我们放弃

代码实现如下

```c
typedef struct{
    ElemType[MAXSIZE];
    int front;
    int rear; //指向第一个空的尾而不是有元素的 根据不同的习惯 记得修正自己的代码
}SqQueue;
Status InitQueue(SqQueue *Q){
    Q->front=0;
    Q->rear=0;
    return OK;
}
int QueueLength(SqQueue Q){
    return (Q.rear-Q.front+MAXSIZE)%MAXSIZE
}
Status EnQueue(SqQueue *Q,ElemType e){
    if((Q->rear+1)%MAXSIZE==Q->front){
        return ERROR;
    }
    Q->data[Q->rear]=e;
    Q->rear = (Q->rear+1)%MAXSIZE;
    return OK;
}
Status DeQueue(SqQueue *Q,ElemType *e){
    if(Q->front == Q->rear){
        return ERROR;
    }
    *e=Q->data[Q->front];
    Q->front = (Q->front+1)%MAXSIZE;
    return OK;
}
//虽然说循环队列是一个非常优秀的存储方式 但是数组真正溢出仍然无法不免 链表结构才是数据结构学科的真正核心
```

### 链式存储结构

我们仍然沿用惯例不舍弃头结点这一链表乃至于后面都有用的重要设计 让front指向头结点而不是真正的队头 rear指向目前的队尾

```c
typedef struct QNode{
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
    s->data = e;
    s->next = NULL;
    Q->rear->next = s;
    Q->rear = s;
    return OK;
}
Status DeQueue(LinkQueue *Q,ElemType *e){
    QueuePtr p;
    if(Q->front==Q->rear){
        return ERROR;
    }
    p = Q->front->next;
    *e = p->data;
    Q->front->next = p->next;
    if(Q->rear == p){
        Q->rear = Q->front;
    }
    free(p);
    return OK;
}
//没有什么难以理解的地方
```

## 串

实际上我们在这一章想要研究的是字符串这一数据类型 很明显这是一个非常有意思的数据类型 因为他契合了现代计算机技术发展以后人们不止在处理数字 而是字符这一趋势 C语言的字符串采用的是字符数组的方式进行存储 很明显我们从这里开始才开始正式仔细研究字符的处理 然后在更靠后的高级语言中进一步的发展 这里我们只会介绍一些并不复杂的内容 更多的串的内容已经被高级语言进行封装 我们并不需要接触那么多

串的长度 空串 空格串 子串和主串 子串在主串的位置 这些内容我们在C语言基础部分已经有了足够的了解 这里不再赘述

**串的比较**

两个数字的比大小不能再容易 但是串也有大小的关系 我们在只需要英文的环境下ASCII编码格式已经能很好的处理问题了 而进一步的Unicode编码的前256个字符和ASCII编码一样  串的大小关系就是在挨个比较字符的ASCII编码的大小 理解串的比较 我们在后面还会用到

### 串的ADT与存储

串和线性表相似吗？ 相似的 但是串的基本操作和线性表天差地别 我们会关注子串的存在 但是线性表中没有这个概念 由于高级语言已经给我们封装好了关于串的大部分操作 所以下面只是一些介绍性的内容

串一般只用顺序存储的方式  链式存储对于字符串没有任何优点 实际上String类型的数据存储在堆中 系统对他进行动态的分配 堆也可以用malloc和free进行管理

```c
int Index(String S,Strint T,int pos){
    int n,m,i;
    String sub;
    if(pos>0){
        n= StrLength(S);
        m= StlLength(T);
        i = pos;
        while(i<=n-m+1){
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
```

### 朴素的模式匹配算法和KMP模式匹配算法

这两个算法处理的都是子串是否存在和定位的问题 **朴素的模式匹配算法**和前面的Index非常接近 但是要把直接调用的函数改为关于数组的循环和判断 下面是代码表述

```c
//请注意 字符串的第一位也就是0出存储了字符串的长度
int Index(String S,String T,int pos){
    int i = pos;
    int j = 1;
    while(i<=S[0]&&j<=T[0]){
        if(S[i]==T[j]){
            j++;
            i++;
        }
        else{
            i = i-j+2;
            j = 1;
        }
    }
    if(j>T[0]){
        return i-T[0];
    }
    else{
        return 0;
    }
}
//在这个程序设计里面 我们就是最简单的挨个核对主串的每一个子串看看能不能找到完全符合的 
//但是很容易发现 如果原本的字符串里面有一些部分重合的段 这个匹配算法的时间复杂度是非常高的 尤其是原本的比对会被转换成二进制码的时候 
```

**所以我们需要改进 也就是KMP模式匹配算法**  算法的命名是算法开发者名字的缩写

我们得先讲一下算法的原理 因为他不是很直观

KMP算法和核心在于原定子串T的首字母和后面不一致 所以如果完成了一些相等判断 就可以舍弃一些没必要的判断了 也就是主串目前的i值不需要继续按照12345进行递增 可以跳过i值的回溯流程 节约资源**（核心点1 i的值不需要回溯 可以直接沿用不会出问题）**

那么j值呢 很明显的 T串首字符和后面字符的比较是重要的  在完全不重复的时候 j要归回1才能比较完成 但是如果有重复 j就会产生不一样的变化 这个变化和当前字符之前的串的前后缀相似度有关系 我们可以把j值的变化单独研究处规律 他脱离S存在

j值满足如下规律

next[j]=0 j=1时

next[j]=k 前j-1个元素的前缀和尾缀的相似度加一 允许一个元素被选中两次 如ababaa 的next[6]=3

next[j] = 1 其他的情况 

有了这些我们就可以代码实现了

```c
void get_next(String T,int *next){
    int i,j;
    i = 1;
    j = 0;
    next[1]=0;
    while(i<T[0]){// 还是意味着长度
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
    while(i<=S[0]&&j<=T[0]){
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
    if(j>T[0]){
        return i-T[0];
    }
    else{
        return 0;
    }
}    
//我们只是对原本的朴素匹配算法进行了轻微的调整 就对有很多部分匹配的效率进行了很好的优化 KMP算法还有更多的改进 我们在这里就不再叙述了 串这里引入算法部分只是希望我们有一个初步的理解 而非全部 这也是这里能做的全部了
```

## 树

树是一种满足一对多的数据结构类型 由于他的特性 我们学会这个结构可以处理很多编程中的问题 

**树TREE是n个结点的有限集 n=0称为空树 任何的一颗非空的树中有且仅有一个特定的结点是树的根Root结点 根的各个子结点又是一颗独立的树 称为子树 SubTree 示意图我们在数据结构的最开篇就已经聊过了**

树的根节点是唯一的 他和现实并不完全一样 子树的个数没有限制他们他们不能够相交 树是一个全新的数据结构的提出 我们会引入很多新的概念

结点拥有的子树的个数称为他的度 Degree

度为0的结点称为叶结点 Leaf

度不为0的结点称为非终端结点或者分支结点 除根节点以外 分支结点也称为内部结点 

树的度是树内各个结点度的最大值

结点子树的根称为这个结点的孩子 Child 反过来 这个结点是他孩子的双亲Parent 同一个双亲的孩子之间称为兄弟Sibling 结点的祖先是从根到该结点的所有结点 反正称为结点的子孙

树存在层次 Level 的概念 根是第一层 他的子孙是第二层 不断类推 双亲在同一层的结点互为堂兄弟 最大层次称为树的深度Depth

如果子树有从左到右的顺序 不能互换 则称为他是有序树 反之称为无序树

森林Forest是多棵互不相交的树的集合 实际上结点的子树就构成森林 

很明显树这一结构和线性表有非常多的不同  而且复杂度大大的提升了 

树的ADT我们在这里就不再赘述了 实际上我们可以根据自己的需求进行设计 

### 树的存储结构

简单的顺序存储结构不能实现对树的存储 我们会结合前面的存储方法设计用于树的存储的结构与方法  很明显的是 我们不可能让结点存储动态的指针域 所以树的设计存在一些受限的可能 没有万能的树 后面很快就会理解这句话

#### 双亲表示法

在双亲表示法中 我们用顺序的数组存储所有的结点 每个结点的指针域都是用来指向他的双亲的 设计如下

```c
typedef struct PTNode{
    ElemType data;
    int parent;
}PTNode;
typedef struct{
    PTNode nodes[MAX_TREE_SIZE];
    int r,n; //这个变量用来存储根节点的位置和结点的数目
}PTree; 
```

对于没有双亲的结点（根节点） 我们只需要要求他的双亲的位置时负一就可以了 在这个设计方法里面 我们寻找孩子是比较困难的 需要遍历整个结构才可以实现（找到双亲是我们需要的结点）当然为了处理这一点 我们可以改进一下 比如添加长子域 没有孩子的结点长子域是负一就好了 修改如下

```c
typedef struct PTNode{
    ElemType data;
    int parent;
    int firstchild;
}PTNode;
```

很明显的这样无法处理研究兄弟的情况 所以我们不妨增加兄弟域

```c
typedef struct PTNode{
    ElemType data;
    int parent;
    int firstchild;
    int rigthsib;
}PTNode;
```

存储结构的设计具有非常大的灵活性 是否继续增加取决于需求而非其他 这是程序设计人员自行决定的

#### 孩子表示法

由于树的结点可能有多棵子树 所以我们考虑使用多重链表 每个结点有多个指针域分别指向子树的结点 由于孩子的数量是不同的 所以不妨在结点中存储他的度来方便我们实现 但是这样产生了结点结构不同的问题 为了处理这个不同 **我们选择把每个结点的孩子结点排列起来 用单链表为存储结构 也就是n个结点就有n个孩子链表** 如果是叶子结点就为空 最后**把这些单链表的头指针存储为一个线性表 这个线性表往往采用顺序存储结构储存（链式也行）**这就是孩子表示法 

```c
typedef struct CTNode{
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
```

有了这样的标识方法 查找结点的孩子和查找结点的兄弟都是比较容易的 但是查找双亲比较困难 我们当然可以结合双亲表示法的好处 形成**双亲孩子表示法**

```c
typedef struct{
    ElemType data;
    ChildPtr firstchild;
    int parent;
}CTBox;
```

#### 孩子兄弟表示法

前面我们是从结点的双亲和孩子入手进行研究 这里尝试一下从双亲的兄弟进行表示 当然 只研究兄弟是无法形成树这样的结构的 我们添加长子 也就是这里我们的重点是firstchild and rigthsib 

```c
typedef struct CSNode{
    ElemType data;
    struct CSNode *firstchild,*rigthsib;
}CSNode,*CSTree;
```

没有向我们的最开篇一样选择数组 因为指针其实很大程度的增加了结构设计的灵活性 这样构成的一个树只是不方便访问双亲 其余非常舒服 实际上这个表示法最大的好处是**把原本复杂的树结构变成了一棵二叉树** 在舍弃了部分结点的前提下 这给了我们非常舒服的性质  （有双亲指针不适用还是不耽误二叉树的存在）

这是最好用的表示方法了 但是实际上 按需设计数据结构才是设计人员需要做到的

### 二叉树定义与性质

**二叉树 Binary Tree** 是n个结点的有限集合 他有一个根节点和两棵互不相交的 分别称为根节点的左子树和右子树的二叉树组成 

* 二叉树每个结点最多有两个子树 可以为1或者0
* 左子树和右子树是有顺序的
* 即便只有一棵树 也区分左子树和右子树
* 他的基本形态分别为
  * 空二叉树
  * 只有一个根结点
  * 根节点只有左子树
  * 根节点只有右子树
  * 根节点有左子树也有右子树

下面介绍一些特殊的二叉树

* 斜树 所有的结点都只有左子树或者右子树  实际上线性表就是特殊的斜树
* 满二叉树 所有的分支结点都存在左子树和右子树 所有的叶子都在同一层上
* 完全二叉树对一棵有n个结点的二叉树按照层序编号 如果编号为i的结点和同样深度的满二叉树比编号为i的结点位置完全相同 他就是完全二叉树 
  * 叶子只在最下面两层
  * 最下层叶子一定在左侧连续位置
  * 倒数第二层的叶子结点一定在右侧连续位置
  * 不存在只有右子树的结点
  * 同样结点的二叉树 完全二叉树深度最小

**二叉树的性质**

* 第i层最多2^(i-1)个结点 因为不会大于满二叉树的结点数目
* 深度为k的二叉树最多拥有2^k-1个结点
* 任意二叉树 如果终端结点n0个 度为2的结点n2个 则 n0 = n2 +1
* n个结点的完全二叉树的深度为 [log~2~n]+1
* 对于一棵有n个结点的完全二叉树 按照层序对结点进行编号 对于任意的结点i
  * i=1时 他是根节点 没有双亲 i>1的时候双亲是 [i/2]
  * 若2i>n 则结点i没有左孩子（他是叶子结点） 否则i的左孩子是2i
  * 若2i+1>n 则结点i没有右孩子 否则i的右孩子是2i+1

### 二叉树的存储，遍历，生成

前面说过了 树这一特殊结构不方便使用顺序存储结构  但是二叉树由于特殊性 所以也可以考虑  我们在这里不研究二叉树的顺序存储结构 而是更关注我们更常用的手段 也就是 **二叉链表**

```c
typedef struct binode
{
    ElemType data;
    struct binode * lchild;
    struct binode * rchild;
}binode,*BiTree;
//实现存储是非常自然的 树指针自然的指向他的孩子 然后一直重复这个过程
```

二叉树的遍历肯定是二叉树使用非常重要的部分 他的核心就是有次序的访问二叉树的所有结点 保证所有的结点都只被访问一次 一共有四种常用的遍历方法 都会在后面被我们讲解 他们的本质都是把二叉树进行线性化来处理 正如同二叉树采用递归原理定义 我们要用递归的方式遍历 理解递归的思想是这里重要的一环

```c
//前序遍历递归二叉树算法
void PreOrderTraverse(BiTree *T)
{
    if(T==NULL)
    return;
    printf("%c", T->data);          //显示结点数据，可以更改为其他对结点操作
    PreOrderTraverse(T->lchild);    //再先序遍历左子树
    PreOrderTraverse(T->rchild);    //最后先序遍历右子树
}
//中序遍历递归二叉树算法
void InOrderTraverse(BiTree *T)
{
    if(T==NULL)
    return;
    InOrderTraverse(T->lchild); //中序遍历左子树
    printf("%c", T->data);      //显示结点数据，可以更改为其他对结点操作
    InOrderTraverse(T->rchild); //最后中序遍历右子树
}
 
//后序遍历递归二叉树的算法
void PostOrderTraverse(BiTree *T)
{
    if(T==NULL)
    return;
    PostOrderTraverse(T->lchild);   //先后序遍历左子树
    PostOrderTraverse(T->rchild);   //再后续遍历右子树
    printf("%c", T->data);  //显示结点数据，可以更改为其他对结点操作
}
//差异其实就是打印位置的不同 本质上思路完全一样
```

实际上生成二叉树就是遍历思想的轻微改变 我们给出前序（更好理解）的序列顺序 反过来填回树里面的思路

```c
//前序遍历递归法建立二叉树算法
void CreatBiTree(BiTree *T)
{
    char data;
    scanf("%c",&data); 
    if(data=='#'){
        T=NULL;        
    }
    else
    {
        *T=(BiTree *)malloc(sizeof(BiTree));
        (*T)->data=data;
        CreatBiTree(&(*T)->lchild);
        CreatBiTree(&(*T)->rchild);
    }
}
```

```c
//按层遍历递归二叉树算法 补充前面没提到的一个内容
void Layer_order(BiTree * TNode,BiTree ** F,BiTree ** R)  //二级指针
{
 
    *F=TNode;            //将当前节点放入队列首指针所指位置
    printf("%c",(*F)->data);
    if((*F)->lchild!=NULL)
    {
    R=R+1;
    *R=(*F)->lchild;    //节点的左儿子放入队尾
    }
    if((*F)->rchild!=NULL)
    {
    R=R+1;                //首指针向后移动一格
    *R=(*F)->rchild;      //节点的右儿子放入队尾
    }
 
    if(F!=R)
 
    {
     F=F+1;
     Layer_order(*F,F,R);//递归
    }
}
```

增加指向双亲的结点构成三叉链表也是非常优秀的方法

### 线索二叉树，二叉树的转换 

线索二叉树存在的意义是研究在某种次序进行遍历的时候，每个结点的前驱和后继 由于线索化本身只是在节约空间 所以我们在这里对于线索二叉树和线索化不详细叙述 以后用到的时候我们再聊这个

如果所有的树和森林都是二叉树 研究起来肯定会非常的方便 前面我们借助孩子兄弟表示法把一颗树进行了二叉链表化 从某种抽象意义上讲 二叉链表就是二叉树的另一种表示形式 所以设定一定的规则 我们能把原本的树甚至是森林二叉树化

对于普通的树 孩子兄弟表示就能够把它二叉树化 每个结点都指向他的长子和右兄弟 只要重新整理顺序就能形成二叉树；

对于森林 每一棵树都是兄弟 所以我们先把他们分别二叉树化  第一棵二叉树不动 很明显的我们可以把其余的树当做右孩子依附到原本的第一棵二叉树上 就能实现二叉树化了

反过来的过程也是并不复杂的 本质上原理是完全一样的 

对于树和森林的遍历  树的遍历和我们刚才二叉树的遍历方法完全一样 原本的树选好孩子兄弟表示法就可以了 森林的遍历就是把所有的树遍历一遍  他们都没有什么复杂的地方

### 赫夫曼树 

赫夫曼编码是我们最常用的压缩编码之一 在保证原始文件不丢失太多观看精度的同时压缩文件体积对于现代计算机的发展非常重要

赫夫曼编码是数学家赫夫曼发明的 他在编码过程中运用了特殊的二叉树 赫夫曼树 我们会在这里简单聊一聊它

我们需要引入树的路径长度的概念 他表示两个结点之间需要判断的次数 有了他我们就能得到树的路径长度（所有结点的路径长度的和）如果我们再给每一个结点一个加权值 就能得到带权路径长度WLP WLP最小的树就称为赫夫曼树  

现在虽然我们知道了赫夫曼树的存在 但是如何建立他呢？

1. 所有结点分别带上他们的权值构成一些根结点（也是一棵树）
2. 在这个集合中选取两个权值最小的结点（树）构造出一棵二叉树 这棵树的权值是两个子树的权值的和 把这棵树加入前面的集合里面 删除原本的树
3. 重复二的过程 直到树变成一棵 

到此为止赫夫曼树的构造就完成了

赫夫曼树并非我们的目的 事实上赫夫曼编码就是根据赫夫曼树形成的 根据左右方向选择0或者1的编码（左0右1）这样就能让出现频次更高的字母有着更小存储空间占用 这就是数据压缩 至于赫夫曼编码压缩在实际使用会出现的问题 我们这里就不叙述了 使用的时候自然会理解

## 图

图 Graph 和我们在拓扑学以及图论中间研究的内容有很大的相似之处 通常表示为G（V,E）分别表示顶点以及边的集合 

在图部分的学习中 我们用顶点 Vertex 来描述结点 正如我们在线性表部分借助元素这个概念一样 理清这些概念上的你内容是我们学习的重要一部分 我们明确的给出要求 顶点几何不能为空 也就是图一定有顶点 所谓的边其实是描述顶点之间的关系的

* **无向边** 顶点之间的边没有方向的区别 **有向边**恰恰和他相反 Undirected Edge 
* 类似的我们有无向图和有向图的概念 
* 对于有向边我们把他的边Edge称为弧Arc 
* 为了方便描述 无向边采用圆括号描述 有向边采用方括号描述
* 如果不存在顶点到他自身的边 并且同一条边不重复出现 则称他为简单图
* 在无向图中 如果所有顶点都有边相连接 则称他是完全无向图 他有 C~n~^2^条边
*  在有向图中  如果所有顶点都有两条方向相反的边相连接 则称他是完全有向图 2*C~n~^2^
* 由于边的数量不定 所以存在稀疏图和稠密图的概念 他没有定量的标准
* 有的边和弧存在相关的数字 称之为权 Weight 带有权的图称为 网 Network
* 子图的概念也是存在的
* 引入邻接点Adjacent的概念 以及 边依附incident与两个邻接点 
* 我们很容易发现 对于无向图 边的数目就是各顶点度数和的一半
* 对于有向图 边的数目等于出度（Outdegree）的和等于入度的和
* 从一个顶点到另一个顶点 我们存在路径Path的概念 路径的长度就是边的数目
* 存在回路Cycle的概念 也称为为环或者回环
* 在无向图中 如果任意选择的顶点都有路径连接 我们称他为连通图 Connected Graph 连通是后面需要研究的非常重要的一个概念
* 无向图的极大连通子图称为连通分量
* 在有向图中 如果都存在双向的连接路径则成为强连通图 对应的有极大强连通子图 强连通分量
* 然后我们再来聊一下生成树 连通图的生成树是极小的连通子图 把包含图中全部n个顶点 但是只有构成一棵树的n-1个边 多余n-1条边一定会成环 有n-1条边也不一定是生成树
* 如果一个有向图恰好有一个顶点入度为0 其余顶点入度为1 则就是一棵有向树 生成森林也是同理的 有向图更好理解一些

### 图的ADT和存储结构

图的ADT就更加复杂了 我们有着常用的各个方法并且根据实际使用的需求随时准备继续添加 在这里不浪费笔墨

非常明显 由于图的多对一和一对多特性 使用顺序存储结构是绝对不现实或者说不可行的 使用多重链表可以实现图结构 但是会浪费很多的指针域 所以我们会使用一些比较有趣的方法存储图结构 正如树结构一样

#### 邻接矩阵 Adjacency Matrix

这个方法的核心思想就是把顶点和边分开存储 顶点使用一维数组就可以很好的解决 边可以使用二维数组 也叫作邻接矩阵就可以处理了

如果两个结点之间有边 就让邻接矩阵的值为1 没有边就是0 对于有向图只需要变换指示符号就可以了 很明显的 简单图的邻接矩阵对角线全为0  并且无向时他是一个对称矩阵 有向图则没有这个性质 

有了邻接矩阵我们很容易能够存储一个图

在前面我们有网的概念 还是更改表示符号添加权值就可以实现 下面是一些代码示例

```c
typedef struct{
    VertexType vexs[MAXVEX];
    EdgeType arc[MAXVEX][MAXVEX];
    int numVertexes,numEdges;
}MGraph;
void CreateMGraph(MGraph *G){
    int i,j,k,w;
    scanf("%d%d",&G->numVertexes,&G->numEdges);
    for(i=0;i<G->numVertexes;i++){
        scanf("%d",&G->vexs[i])
    }
    for(i=0;i<G->numVertexes;i++){
        for(j=0;j<G->numVertexes;j++){
            G->arc[i][j] = 65535;//这是初始化的意思 权值不可能为这个值
        }
    }
    for(k=0;k<G->numEdges;k++){
        scanf("%d%d%d",&i,&j,&w);
        G->arc[i][j] = w;
        G->arc[j][i] = w;        
    }//识别与创建部分
}
//很明显能看到 这个时间复杂度在n^2级别 并不算低
```

#### 邻接表 Adjacency List

很明显的 领接图的矩阵在二维矩阵上面耗费了非常大的存储空间 并且当稀疏图的时候 大量的存储空间没有发挥作用 我们考虑使用线性表来节约存储空间 Adjacency List  这个思路和我们在树部分的前两个方法很像

顶点采用一维数组进行存储 当然链表也可以 顶点数组也需要存储数据元素指向第一个邻接点的指针  我们让每一个顶点的所有邻接点构成一个线性表  来实现存储空间的节约 带有权值的话就在线性表的每一个元素上面田间权值相关变量 有向的话其实原理是完全一样的只是凭空增加了一个逆邻接表的概念  下面是代码示例

```c
typedef struct EdgeNode{
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
    scanf("%d%d",&G->numVertexes,&G->numEdges);
    for(i=0;i<G->numVertexes,i++){
        scanf("%d",&G->adjList[i].data);
        G->adjList[i].firstedge=NULL;
    }
    for(k=0;k<G->numEdges,k++){
        scanf("%d%d",&i,&j);
        e = (EdgeNode*)malloc(sizeof(EdgeNode));
        e->adjvex=j;
        e->next = G->adjList[i].firstedge;
        G->adjList[i].firstedge = e; //下面还要创建对等的
        e->adjvex=i;
        e->next = G->adjList[j].firstedge;
        G->adjList[j].firstedge = e; 
        //这里的链表用的是头插法 前面已经介绍过了 逆邻接表和有向的和这个没有本质的区别
    }
}
```

#### 十字链表

十字链表的本质就是结合了邻接表和逆邻接表 便于我们同时研究出度和入度 结构变化如下

```c
typedef struct VertexNode{
    ElemType data;
    EdgeNode *firstin;
    EdgeNode *firstout;
}VertexNdoe,AdjList[MAXVEX];
```

他最大的用处就是处理有向图 因为只有这时候才要细节研究出入的度

#### 邻接多重链表

邻接多重表是一种对无向图的优化结构 他的目的是吧边表中两个结点才能描述一个边进行优化 我们在这里不进行额外的叙述 **理解邻接表是我们学习的核心一步**

#### 边集数组

边集数组就是更简化的一个方法了 把顶点的信息存储在一个数组里面 边的信息存储在一个数组里面 他不适合常见的删除 添加 查找操作 只是适合遍历所有边来实现某个目的 我们在后面为提到一些相关的算法

### 图的遍历 TraversingGraph

很明显我们需要研究这个 但是图可比树复杂多了 在树的遍历中我们借助二叉树进行研究 并且用孩子兄弟表示法把原本的树二叉树化 但是在复杂的图里面 我们没有这个技巧 唯一能做的就是在走过的路上刻下记号 来避免重复

除了重复以外 图的遍历只有一个核心要素了 不要走入死路  这就是深度优先和广度优先 

有向图研究遍历和无向没有任何区别 单向连通也是连通

#### 深度优先搜索 DFS

depth_first_search 本质就是一个递归的过程 他的原理有点类似于我们前面的前序搜索 他不断的访问标记直到找不到没有被标记的元素 然后就回退 继续寻找 最后层层函数返回到最初的时候 我们就完成了这个连通图的遍历/搜索 如果有还没有被标记的结点 肯定不在这个图里面 需要再进行DFS 在邻接表里面 按照邻接顺序就是最简单的固定顺序了

```c
void DFS(GraphAdjList GL,int i){
    EdgeNode *p;
    visited[i] = TRUE;
    printf("%c",GL->adjList[i].data);
    p = GL->adjList[i].firstedge;
    while(p){
        if(!visited[p->adjvex]){
            DFS(GL,p->adjvex);
        }
        p = p->next;
    }
}
void DFSTraverse(GraphAdjList GL){
    int i;
    for(i=0;i<GL->numVertexes;i++){
        visited[i] = FALSE;
    }
    for(i=0;i<GL->numVertexes;i++){
        if(!visited[i]){
            DFS(GL,i);   //如果图是连通的 这个DFS只会执行一次 
        }
    }
}
```

#### 广度优先搜索 BFS

广度优先就是一个分层的手段了 类似层序遍历的手对岸 先距离为1的 标记 然后研究距离为2的 继续标记  一直循环 所有需要出队的元素都会被检查邻接点 这就是BFS算法的核心 两种算法的时间复杂度是完全一样的 前面的层序遍历对于理解BFS还是比较重要的 两种算法都要求掌握

```c
void BFSTraverse(GraphAdjList GL){
    int i;
    EdgeNode *p;
    Queue Q;
    for(i=0;i<GL->numVertexes;i++){
        visited[i] = FALSE;
    }
    InitQueue(&Q);
    for(i=0;i<GL->numVertexes;i++){
        if(!visited[i]){
    		visited[i] = TRUE;
            printf("%c",GL->adjList[i].data);
            EnQueue(&Q,i);
            while(!QueueEmpty(Q)){
                DeQueue(&Q,&i);
                p = GL->adjList[i].firstedge;
                while(p){
                    if(!visited[p->adjvex]){
                        visited[p->adjvex]=TURE;
                        printf("%c",GL->adjList[p->adjvex].data);
                        EnQueue(&Q,p->adjvex);
                    }
                    p=p->next
                }
            }
        }
    }
}
```

### 最小生成树

我们提到过 连通图的生成树是他的极小连通子图 他含有全部的顶点 但是只有n-1个边 我们把构造连通网最小代价（很明显是有权值的）生成树称为最小生成树 Minimum Spanning Tree

我们在这里介绍两个关于生成最小生成树的算法 请注意 负边权和不连通图考虑最小生成树是不明智的

#### prim算法

有点类似于一个前面的染色的过程 选择边权最小的顶点进行染色 

```c
void MiniSpanTree_Prim(MGraph G){  //我们选择矩阵形式
    int min,i,j,k;
    int adjvex[MAXVEX];
    int lowcost[MAXVEX];
    lowcost[0]=0; //到集合的距离
    adjvex[0]=0;
    for(i=1;i<G.numVertexes;i++){
        lowcost[i] = G.arc[0][i];
        adjvex[i] = 0;
    }
    for(i=1;i<G.numVertexes,i++){
        min = INF;
        j=1;
        k=0;
        while(j<G.numVertexes){
            if(lowcost[j]!=0&&lowcost[j]<min){
                min = lowcost[j];
                k=j;
            }
            j++;
        }
        printf("(%d,%d)",adjvex[k],k);
        lowcost[k]=0;
        for(j=1;j<G.numVertexes;j++){
            if(lowcost[j]!=0&&G.arc[k][j],lowcost[j]){
                lowcost[j]=Garc[k][j];
                adjvex[j]=k;
            }
        }
    }
}
```

#### Kruskal算法

在这里我们是从边入手 而非前面的顶点入手 找到那些权值最小的边是这里的核心 很明显这个算法需要判断选边以后会不会出现环路 这就是他最重要的地方了 我们采用标记顶点的方法 如果两个顶点有相同标记再连接起来就是环路结构了

为了方便后面 这里采用边集数组的方法 转化矩阵表示到边集数组不是很困难的事情

```c
void MiniSpanTree_Kruskal(MGraph G){
    int i,n,m;
    Edge edges[MAXEDGE]; //按照权值排好顺序的代码我们省略了
    int parent[MAXVEX];
    for(i=0;i<G,numEdges;i++){
        parent[i]=0;
    }
    for(i=0;i<G.numEdges,i++){
        n=Find(parent,edges[i].begin);
        m=Find(parent,edges[i].end);
        if(n==0||m==0){
            parent[edges[i].begin]=1;
            parent[edges[i].end)]=1;
            printf("(%d %d) %d",edges[i].begin,edges[i].end,edges[i].weight)
        }
    }
}
int Find(int *parent,int f){
    while(parent[f]!=0){
        return 1;
    }
    return 0;
}
```

非常明显的 两个算法一个针对边比较多的情况 一个针对边比较少的情况 合适的选择是我们的核心 理解算法的思想 并且写出自己的实现才是要点

### 最短路径问题

#### Dijkstra算法

这是选择按照路径长度递增来选取最短路径的算法，是从BFS算法的一种改进，通过不断的扩展搜索最近能找到的节点。

```c
int Pathmatirx[MAXVEX];  //前驱顶点的下标 实际上就是路径数组 它存储的是路怎么走
int ShortPathTable[MAXVEX]; //最短路径的存储
void ShortestPath_Dijkstra(MGraph G,int v0,Pathmatirx *P,ShortPathTable *D){
    int v,k,w,min;
    int final[MAXVEX]; //存储这个顶点有没有找到最短路径的状态 1就是找到了 0是没找到
    for(v=0;v<G.numVertexes;v++){
        final[v]=0;
        (*D)[v] = G.matirx[v0][v];
        (*P)[v] = 0;
    }
    (*D)[v0] = 0; 
    final[v0] = 1; //第一个顶点的路径已经确定 到本身
    for(v=1;v<G.numVertexes;v++){
        min = INF;
        for(w=0;w<G.numVertexes;w++){
            if(!final[w]&&(*D)[w]<min){
                k = w;
                min = (*D)[w];
            }
        }
        final[k] = 1;
        for(w=0;w<G.numVertexes;w++){
            if(!final[w]&&(min+G.matrix[k][w])<(*D)[w]){
                (*D)[w] = min+G.matrix[k][w];
                (*P)[w] = k;
            }
        }
    }
}
```

#### Floyd算法

这个算法是同时考虑了所有点和所有点之间的最短路径 让所有点都尝试一下作为中继看看能不能优化算法

```c
int Pathmatirx[MAXVEX][MAXVEX];
int ShortPathTable[MAXVEX][MAXVEX];
void ShortestPath_Floyd(MGraph G,Pathmatrix *P;ShortPathTable *D){
    int v,k,m;
    for(v=0;v<G.numVertexes;v++){
        for(w=0;w<G.numVertexes;w++){
            *D[v][w] = G.matrix[v][w];
            *P[v][w] = w;               
        }
    }
    //初始化的过程
    for(k=0;k<G.numVertexes;k++){
        for(v=0;v<G.numVertexes;v++){
            for(w=0;w<G.numVertexes;w++){
                if((*D)[v][w]>(*D)[v][k]+(*D)[k][w]){
                    (*D)[v][w]=(*D)[v][k]+(*D)[k][w];
                    (*P)[v][w]=(*P)[v][k]
                }
            }
        }
    }
    //这个直接对原始的修正是非常巧妙的 很简洁的代码实现了很复杂的功能
}
```

### 拓扑排序

我们说完了两个有环的应用（最短路径和最小生成树） 现在来想想没有环的应用 也就是拓扑排序的问题 他的本质是一种无向的活动规划图 很明显活动是不能举办回去的 有的条件不完成后续也没办法干 所以我们引入了AOV网 也就是 Activity On Network 在处理关于AOV网的问题的时候 我们引入了拓扑序列的概念 对于有向图 如果序列路径不能反过来 他就是拓扑序列 

所谓的拓扑排序 就是对一个有向图构造拓扑序列的过程 构造的时候如果全部的顶点都被输出了 证明肯定不存在回路 否则他不是AOV网 很明显的对于工程顺序的安排就是实现拓扑排序的过程

拓扑排序的基本思路 选择入度为0的点输出 删除这个顶点和以他为尾的弧 重复这个过程 

为了方便删除 我们选择邻接表来实现我们的需求 我们要建立一个方便我们这里使用的邻接表 他要着重研究入度 反而出度不是很重要

```c
/* 拓扑排序，若GL无回路，则输出拓扑排序序列并返回OK，若有回路返回ERROR */
Status TopplogicalSort(GraphAdjList GL){
	EdgeNode *e;
	int i, k, gettop;
	int top = 0;	//用于栈指针下标
	int count = 0;	//用于统计输出顶点的个数
	int *stack;	//建栈存储入度为0的顶点
	stack = (int *)malloc(GL->numVertexes * sizeof(int));
	for(i=0;i<GL->numVertexes;i++)
		if(GL->adjList[i].in == 0)
			stack[++top] = i;	//将入度为0的顶点入栈
	while(top != 0){
		gettop = stack[top--];	//出栈
		printf("%d->",GL->adjList[gettop].data);	//打印此顶点
		count++;		//统计输出顶点数
		for(e = GL->adjList[gettop].firstedge;e;e = e->next){	//对此顶点弧表遍历
			k = e->adjvex;
			if(!(--GL->adjList[k].in))	//将k号顶点邻接点的入度减1
				stack[++top] = k;	//若为0则入栈，以便下次循环输出
		}	
	}
	if(count <GL->numVertexes)	//如果count小于顶点数，说明存在环
		return ERROR;
	else
		return OK;
}
```



### 关键路径

这个问题和拓扑排序有点类似 他们都是没有环的应用 都需要类似的思路来安排工程 但是关键路径问题需要研究时间 如果AOC网的带上时间权值 他就变成了AOE网 此时我们不是在安排工作的顺序 而是希望研究一下时间 从源点到汇点的最长时间路径被我们称为关键路径 上面的活动称为关键活动 想要提高整体效率就要从关键活动入手 很明显的关键路径算法和我们的拓扑排序有很大的相似之处 

几个关键参数

事件的最早发生时间etv：即顶点vk的最早发生时间；
事件的最晚发生时间ltv：即顶点vk的最晚发生时间，也就是每个顶点对应的事件最晚需要开始的时间，超出此时间将会延误整个工期；
活动的最早开工时间ete：即弧ak的最早发生时间；
活动的最晚开工时间lte：即弧ak的最晚发生时间，也就是不推迟工期的最晚开工时间。
可以由1和2求的3和4，然后再根据ete[k]是否与lte[k]相等来判断ak是否是关键活动

```c
int *etv, *ltv;	//事件最早发生时间和最迟发生时间数组
int *stack2;	//用于存储拓扑序列的栈
int top2;	//用于stack2的指针
//改进的拓扑排序代码 用于研究关键路径问题
Status TopologicalSort(GraphAdjList GL){
	EdgeNode *e;
	int i, k, gettop;
	int top = 0;	//用于栈指针下标
	int count = 0;	//用于同级输出顶点的个数
	int *stack;	//建栈将入度为0的顶点入栈
	stack = (int*)malloc(GL->numVertexes * sizeof(int));
	for(i=0;i<GL->numVertexes;i++)
		if(0 == GL->adjList[i].in)
			stack[++top] = i;
	top2 = 0;	//初始化为0
	etv = (int*)malloc(GL->numVertexes*sizeof(int));	//事件最早发生时间
	for(i=0;i<GL->numVertexes;i++)
		etv[i] = 0;	//初始化为0
	stack2 = (int*)malloc(GL->numVertexes*sizeof(int));	//初始化
	while(top!=0){
		gettop=stack[top--];
		count++;	
		stack2[++top2] = gettop;	//将弹出的顶点序号压入拓扑序列的栈
		for(e=GL->adjList[gettop].firstedge;e;e=e->next){
			k = e->adjvex;
			if(!(--GL->adjList[k].in))
				stack[++top] = k;
			if((etv[gettop]+e->weight > etv[k])	//求各顶点事件最早发生时间值
				etv[k] = etv[gettop] + e->weight;	//前一个结点得权值加上当前边的权值，如果大于当前结点已经得到的权值，那么替换，得到当前结点最早发生时间值
		}
	}
	if(count < GL->numVertexes)
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
	ltv = (int*)malloc(GL->numVertexes*sizeof(int));	//事件最晚发生时间
	for(i=0;i<GL->numVertexes;i++)
		ltv[i] = etv[GL->numVertexes-1];	//初始化ltv，初始化为最后那个结点的最早开始时间
	while(top2 != 0){	//计算ltv
		gettop = stack2[top2--];	//将拓扑序列出栈，后进先出
		for(e=GL->adjList[gettop].firstedge;e;e=e->next){	//求各顶点事件的最迟发生时间ltv值
			k = e->adjvex;
			if(ltv[k]-e->weight < ltv[gettop])	//求各顶点事件最晚发生时间ltv，其中，gettop点为k结点的前一个结点.ltv[gettop]为已知的该结点最晚发生时间
				ltv[gettop] = ltv[k] - e->weight;
		}
	}
	for(j=0;j<GL->numVertexes;j++){	//求ete，lte和关键活动
		for(e=GL->adjList[j].firstedge;e;e=e->next){
			k=e->adjvex;	//拿到邻接点下标
			ete = etv[j];	//活动最早发生时间
			lte = ltv[k] - e->weight;	//活动最迟发生时间
			if(ete == lte)	//两者相等即在关键路径上
				printf("<v%d,v%d> length:%d,",GL->adjList[j].data, GL->adjList[k].data, e->weight);
		}
	}
}
//如果有多个关键路径 影响一条是无效的 徐涛对多条关键路径下手
```
