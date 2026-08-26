---
title: 'C Programming Basics: Types, Control Flow, Arrays, and Pointers'
title_zh: C 语言程序设计基础：类型、控制流、数组与指针
date: 2023-03-23 23:35:43 +0800
categories:
- Programming
- C & C++
tags:
- C
author: Hyacehila
mathjax: false
hidden: true
excerpt: Covers language overview, compilation environments, keywords, data types, operators, statements, loops, arrays, functions,
  and pointers.
description: Covers language overview, compilation environments, keywords, data types, operators, statements, loops, arrays,
  functions, and pointers.
excerpt_zh: 整理语言概览、编译环境、关键词、数据类型、运算符、语句、循环、数组、函数和指针。
permalink: /blog/2023/03/23/c-programming-basics-learning-notes/
lang: en
translation_key: 2023-03-23-c-programming-basics-types-control-flow-pointers
translation_status: machine
translation_source_hash: cbd6b32df2a718ba2a54e6d8fc1f0231952432fa9bccf88f4a00c879a2681111
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Overview</h2>
<p>From here, we begin to learn the basic programming language, C, as a more based language, without complex object-oriented work and with the simplest possible grammatical rules, so from here it's a good idea to start with.</p>
<h3>Origin of C language</h3>
<p>The C language is the language that the programmer designed for, and based on, B language, which is now one of the most important languages, understanding C language can understand more deeply the logic of computer bottom operation.</p>
<h3>Reasons for using the C language</h3>
<h4>Control properties</h4>
<p>This design allows users to plan the overall structure from top down.</p>
<h4>Efficiency</h4>
<p>Because of the precision control that is available in the lower C language, we can make the program work more efficiently.</p>
<h4>Portability</h4>
<p>The language of the insemination is closely related to the processor.</p>
<h4>Flexibility</h4>
<p>A lot of operating systems and compilers are written in C, and understanding C is a low-level logic for many languages.</p>
<h4>For programmers</h4>
<p>C language allows us access to hardware, to operate memory, and so much freedom is required by programmers.</p>
<h4>Basis for other programming languages</h4>
<p>In C language, we will learn the basic thinking and logic of programming, starting with a simple language.</p>
<h4>Disadvantages</h4>
<p>The excess flexibility naturally entails a greater risk, and the higher operational efficiency means a very detailed modification, and the disadvantages and advantages are actually from the same place.</p>
<h3>Development orientation</h3>
<p>C++ is the way forward for C language, and he added to C the object-oriented operation required for modular programming, and actually almost all C-plus programs can be seen as a C++ program.
The many advantages of C language have allowed him to be used in embedded programming, microprocessor programming, scientific programming, operating system development, etc.</p>
<h3>The basic working principles of computers</h3>
<p>We have already given an important explanation of the computer's workings in the computer composition.
We've only said a million words here to finish it.
It's the age of advanced languages.</p>
<h3>Use of C language</h3>
<ul>
<li>Sets the objectives of the procedure</li>
<li>Design Program</li>
<li>Write Code</li>
<li>Compile</li>
<li>Run</li>
<li>Test and Debug</li>
<li>Maintenance and modification
In the process of compilation, we have experienced a compilation of the languages of compilation and compilation into the machine language, and have finally made our advanced language into a machine language that hardware understands.</li>
</ul>
<h3>Programming mechanism</h3>
<p>Let's go over some of the important processes ahead.
After completing the design C code, the compiler first translated and compiled our target code into the target code file.
And then we'll add the start code to the interface between the program and the operating system, and then we'll add the code from the C-C library we're using, and we'll link the three to the same thing, and we'll create the enforceable file we need.
In the current development environment, both process compilers can work directly with the compilers and links.</p>
<h3>Compiled in several development environments</h3>
<h4>Unix</h4>
<pre><code class="language-c">cc inform.c  //编译指令
ls //列出文件
a.out //发现可执行文件
a.out //执行这个文件
a.o // 出现一个一般会被删除的目标代码文件
</code></pre>
<h4>Linux</h4>
<pre><code class="language-c">gcc inform.c //编译指令
</code></pre>
<h4>Other integrated development environments</h4>
<p>We don't need to operate in the command line interface, and the location of the compilation depends on the design of the Ide.</p>
<h3>Language Standard</h3>
<h4>ANSI C or C90</h4>
<p>He has some basic guidelines.</p>
<ul>
<li>Trust the program designer.</li>
<li>Don't interfere with what the programmer is doing.</li>
<li>Keep your language short.</li>
<li>Just one way to do one.</li>
<li>I'm gonna make it work at the expense of portability.
It's actually a certain spirit of the C language.</li>
</ul>
<h4>C99</h4>
<p>Just a little change.</p>
<ul>
<li>Supporting international programming</li>
<li>Fix and improve his computing utility.
In fact, C99 has kept its original design thinking, and then we'll explain the changes that have occurred when we need them.</li>
</ul>
<p>The questions in this article can also be addressed<a href="/en/blog/2023/03/18/python-basics-learning-notes/">Python Foundation: Syntax: Data Structure and File Processing</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h2>Language Overview C</h2>
<p>The first chapter is just an overview of what we're learning.</p>
<h3>Simple examples</h3>
<pre><code class="language-c">#include &lt;stdio.h&gt;  //引入我们需要的包 stdio 是关于输入输出的基本程序包
int main(void)  //函数的参数和返回值情况说明
{  // 花括号是常用的分隔符号
	int num;  // 声明  C语言要求声明变量
	num = 1;  //赋值
	printf(&quot;I am a simple&quot;);
	printf(&quot;computer.\n&quot;);
	printf(&quot;My favorite number is %d because it is the first.\n&quot;,num);
	//输出一些内容  \n 是换行  %d 是格式占位
	return 0;  // 函数的返回值
	// //符号是C语言最常用的注释符号
}
</code></pre>
<h3>Name of variable</h3>
<p>We should use the names of meaningful variables and functions and not the identification already used in the C language. Arguments
Python Foundation: Some variable naming rules</p>
<h3>Debug and modify</h3>
<p>Debug is an inevitable problem with program design.</p>
<h4>Wrong Category</h4>
<ul>
<li>Syntax error. Compiler can help you analyze.</li>
<li>It's a semantic error. It's only human.</li>
</ul>
<h4>Some DEBUG technique.</h4>
<ul>
<li>Inner brain compilation</li>
<li>Use debugger</li>
<li>Add some output statements to debug</li>
</ul>
<h3>C language keywords and statements</h3>
<p>Do not use keywords already in the program language, which would cause a big problem, and keep language identifiers, such as those whose initial underlined names have already been used, and not use them, and we can't forget him when we name things like functions and variables.</p>
<h4>C90</h4>
<ol>
<li>auto: declare automatic variables;</li>
<li>break: Jump out of the current cycle;</li>
<li>Case: switch statement branch;</li>
<li>char: Declaration of the character variable or function returns the value type;</li>
<li>const: declare read-only variables;</li>
<li>Continue: End the current cycle and start the next cycle;</li>
<li>(b) default branch in the switch statement;</li>
<li>do: Circle of circular statements;</li>
<li>(a) double: declare the double-precision float variable or function return value type;</li>
<li>ELSE: Derogation of conditional statements (in conjunction with if);</li>
<li>enum: declaration of the type of count;</li>
<li>Extern: declare that variables or functions are defined in other files or other locations in this document;</li>
<li>float: Declaration of float-type variables or function returns value type;</li>
<li>For: a circular statement;</li>
<li>goto: Unconditional jumper;</li>
<li>If: Conditional statement;</li>
<li>Int: Declaration of integer variables or functions;</li>
<li>long: Declaration of long integer variable or function return value type</li>
<li>(a) Register: declaration of the register variable;</li>
<li>return: sub-app returns statement (with or without parameters)</li>
<li>Short: Declaration of short integer variables or functions;</li>
<li>Signed: declares that there are symbol type variables or functions;</li>
<li>sizeof: the number of bytes in which the data type or variable is calculated</li>
<li>(b) Static: declaration of static variables;</li>
<li>(c) sruct: declaration of the type of structure;</li>
<li>switch: for switch statements;</li>
<li>typedef: used to identify aliases for data types;</li>
<li>Unsigned: declare a non-signed type variable or function;</li>
<li>(a) Organization: declaration of the type of common body;</li>
<li>(a) void: the declaration function does not return values or parameters and states that it does not have a type pointer;</li>
<li>(a) Volatile: to indicate that variables can be implicitly changed in the execution of the program;</li>
<li>(a) While: circular conditions for circular statements;</li>
<li>Asm: for embedding compilation instructions in C language;</li>
<li>Fortran: a conditional support type command for the Fortran language link;</li>
</ol>
<h4>C99</h4>
<ol>
<li>Inline: The inline function used to define a class is introduced mainly because it replaces the macro definition in C expression;</li>
<li>Restrict: can only be used to qualify and limit the pointer and to indicate that the pointer is the only initial and initial way to access a data object. That is, all changes to the pointer to the contents in the memory must be made through the pointer and not by other means (other variables or pointser), with the advantage of helping the compiler to optimize the code better and generate more efficient compilation codes;</li>
<li>Bool: Boolean type data with values of 0 or 1, which are used primarily to determine whether conditions are valid or not;</li>
<li>Complex: used to indicate the type of complex number;</li>
<li>Imaginary: used to indicate a figment type;</li>
<li>Pragma: Same functionality as the #pragma command;</li>
</ol>
<h4>C11</h4>
<ol>
<li>Alignas: Specify a variable aligned to other data types;</li>
<li>Alignof: specify the number of bytes of the memory matching of the data type;</li>
<li>Atomic: Atom-type description and qualifier;</li>
<li>Static assert: Declares valid at the time of compilation, and it tests software assertions that are specified by the user and can be converted to a boolean value integer expression. If the expression is calculated to be zero (false), the compiler sends the user-designated message and the translation fails due to an error;</li>
<li>Noreturn: Shows that the call completed does not return the main call function, so that the user and compiler are informed that this special function does not return the control to the main caller, that it is used to avoid abuse of the function, and that the notification compiler optimizes some codes.</li>
<li>Thread local: It affects the storage cycle of variables, modified variables have linear cycles, they are generated at the beginning of the online process and destroyed at the end of the online process. And each thread has an example of a single variable. (a) It can be used in conjunction with the stic and extern keywords, which will affect the link properties of the variables;</li>
<li>Generic: A set of functions of different types but with the same function can simply be abstracted as a single interface;
<strong>Controls have been added to the keyword.</strong></li>
</ol>
<h2>Data and C</h2>
<p>In this chapter, we're going to introduce more basic knowledge of the C language, which is very simple.</p>
<h3>Simple examples</h3>
<pre><code class="language-c">#include&lt;stdio.h&gt;
int main(void)
{
	float weight;  //这里我们给出了新的数据类型 float浮点数可以处理小数了
	float value;
	scanf(&quot;%f&quot;,&amp;weight);  //scanf 语句可以提供输入的交互  &amp;符号是用来获得地址的
	value = 770*weight*14;
	printf(&quot;%.2f&quot;,value);
	return 0;
	//声明各种类型的变量的时候就是申请了存储空间并且建立这个变量
}
</code></pre>
<p>We've already given a description of the data types in the keyword. <a href="/en/blog/2023/03/23/c-programming-basics-learning-notes/">C language keywords and statements</a>
So many medium-digit types are used to save storage space when needed, and if the number given exceeds the range allowed by the type, there's a problem.
We'll give a detailed explanation of the data types of binary storage elsewhere.</p>
<h3>Numbers of other progression systems</h3>
<p>Octal data identifier is %o
Hexadecimal numbers correspond to %x and %X
If we want to add # in the numeric, you can show the digit digit digit digit digit digit</p>
<h3>Character</h3>
<h4>Print Character</h4>
<p>We use char as character identifiers, and we usually only use ASCII, Unicode in C languages to show more languages, but it may not be as meaningful as it is in C.</p>
<pre><code class="language-c">char a = &#39;x&#39;;
</code></pre>
<p>It's the basic way of saying it.</p>
<h4>Non-printing Characters</h4>
<table>
<thead>
<tr>
<th>Conversion sequence</th>
<th>Meaning</th>
</tr>
</thead>
<tbody><tr>
<td>\a</td>
<td>Alert.</td>
</tr>
<tr>
<td>\b</td>
<td>Back</td>
</tr>
<tr>
<td>\f</td>
<td>Page Retrieval</td>
</tr>
<tr>
<td>\n</td>
<td>Line Break</td>
</tr>
<tr>
<td>\t</td>
<td>Horizontal Tab</td>
</tr>
<tr>
<td>\v</td>
<td>Tabs Vertically</td>
</tr>
<tr>
<td>\</td>
<td>Backslash</td>
</tr>
<tr>
<td>&#39;</td>
<td>Single quote</td>
</tr>
<tr>
<td>&#39;&#39;</td>
<td>Double Quote</td>
</tr>
<tr>
<td>?</td>
<td>Question mark</td>
</tr>
<tr>
<td>\0oo</td>
<td>Octahedron</td>
</tr>
<tr>
<td>\xhh</td>
<td>Hexadecimal</td>
</tr>
<tr>
<td>\r</td>
<td>Back to the car.</td>
</tr>
</tbody></table>
<h3>Use data type</h3>
<p>The type of error in the translation of data is very tolerant in the C language, but be careful, because formal tolerance makes it easier for C to make mistakes.
The printf and scanf are two special functions that they have no limit on the number of parameters, and all the more careful about the potential errors in this, because the compiler's high probability is not sending a clear hint (the C language has a test for the number and type of function, but the special function does not work) so it's important to ensure that the number of the format description is the same and the number of the subsequent parameters is the same and the type is consistent.
This interactive statement is based on buffer design, i.e., a buffer mechanism, and if the structure of the buffer zone does not refresh the printf statement, it will not be shown on interfaces such as scanf and \n.</p>
<h2>Formatting String Inputs and Output</h2>
<p>We've been talking about integers, floating points, characters, and so on.</p>
<h3>Simple examples</h3>
<pre><code class="language-c">#include&lt;stdio.h&gt;
#include&lt;string.h&gt;
#define DENSITY 62.4
int main
{
	float weight,volume;
	int size,letters;
	char name[40];  //字符数组就是字符串的核心表达方式
	scanf(&quot;%s&quot;,name);
	printf(&quot;%s&quot;,name);
	printf(&quot;AAAAAA&quot;);
	size = sizeof name;
	letters = strlen(name);
	printf(&quot;%d %d&quot;,letters,size);
	return 0;
}
</code></pre>
<p>Here we have a little bit of a note.
The previous use of the C Macro was used to introduce several string-related functions</p>
<h3>String</h3>
<p>We use double quotation marks to identify him as a string, just like a single quote to identify characters, and the character array is the end of a string, a non-printed character, to mark the end, where %s is the amount we give to indicate a string, which means that the character is different from the character array with only one character.
You know, scanf for spaces, tabs, line breaks as end marks, so he's not a good way to read a sentence.
The sizeofspeceofs give the number of bytes of occupancy, which can be used to determine the length, but it requires more consideration by the program designer.
This single-design function is obviously a better option.</p>
<h3>Constants and preprocessors</h3>
<p>The définine macros mentioned earlier are a good way to make constant determinations, and we'll then introduce other methods, whatever they use #
The pre-treatment instructions are not executed by the compiler, but by pre-treatment, which is usually very easy to pre-process.</p>
<pre><code class="language-c">const int A = 100;
</code></pre>
<p>It's a good way to define constants, and right now we'll be wrong if we change A.</p>
<h3>Input and Output Commands</h3>
<p>The core point is that the format is a bit more than a bit more than a bit more than a bit more than a bit more than a bit more than a bit more than a bit more than a bit more than a bit more than a bit more than a bit more than a bit more than a bit more than a bit more than a bit more than a bit more than a bit more than a bit more than a bit more than a bit more than a bit more than a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a bit of a
Well, the printf actually has a return value, and the return is the number of printed variables.
The scanf and his principles are basically the same, and he'll also enter in the format specified.
It's only necessary to understand the formatting of these two instructions.
The format conversion in format is sometimes at the heart of the complexity of the C language, and we need to consider carefully the changes in the format of the variables, especially the automatic conversion of formats in some places, although we think automatic conversions are often done up to the point where no information is lost.
He also returns values if read failed Return 0 if read to end of file Return EOF constant</p>
<h2>Operator Expression Statement</h2>
<p>Here we'll introduce some of the algorithms that are actually easy to understand, and at the same time we'll introduce some of the words that we've never mentioned before.</p>
<h3>♪ While the cycle ♪</h3>
<p>The methodology used is as follows:</p>
<pre><code class="language-c">while(a&lt;b){
	xxxxxxxxxxxx;
}
</code></pre>
<p>And it's very simple to do something that you repeat under certain conditions.
It's because we need to judge whether conditions are established and certain statements are being executed that the operator has meaning.</p>
<h3>Basic Operators</h3>
<h4>Values</h4>
<pre><code class="language-c">a=b;
</code></pre>
<p>The modified variable to the left of the attribute symbol is required to be modified by the left or not.</p>
<h4>Some other basic algorithms</h4>
<pre><code class="language-c">a+b;
a-b;
a=-b;
a*b;
a++;
a/b; // 这是抛弃小数的除法
a%b  //取余数
sizeof()  //字节为单位返回大小
+=;
-=;
*=
/=;
%=;
</code></pre>
<p>The operator has priority, and we just need to know that he's like math, and if there's any way to remember, we can do it in brackets.
The operator is not a show-off. We need to make the code readable.</p>
<h3>Relationship Operators</h3>
<pre><code class="language-c">&lt;;
&gt;;
&lt;=;
&gt;=;
==;
!=;
//浮点数不进行相等判断 存在舍入误差
//关系运算符的结果是真值 0是否 其余非0的值是真
//目前已经开发了_Bool类型 不过在结果上和原本一样 只是变量类型变更
//优先级应当靠增加括号解决
</code></pre>
<h3>Expressions and Statements</h3>
<p>The sum of operators and numbers at the time of expression
Statements are made of expressions and some syntax symbols
They're the key to the process.
Composite statement, which is that multiple statements are contained in one parenthesis, called code Blocks</p>
<h2>Loop</h2>
<p>Here's a complete description of how the cycle is formed and how it's used.</p>
<h3>And then we'll talk about the cycling.</h3>
<p><code>while</code> It is a cycle of entry conditions: the condition is judged first, the condition is implemented on a genuine basis, and the condition-related variables are usually updated within the circle.</p>
<pre><code class="language-c">while (a &lt; b) {
    /* loop body */
}
</code></pre>
<pre><code class="language-c">status = scanf(&quot;%d&quot;,&amp;num);
while(status==1){
	xxxxxxx;
	status = scanf(&quot;%d&quot;,&amp;num);
}
//我们借助scanf返回值实现循环体的判断 这也是常用的手段 节约了再设立一次变量的资源
while(scanf(&quot;%d&quot;,&amp;num)==1){
	;  //在不需要其他操作的时候 这样写是一个常见的写法
}
//这是另一个常用的简化写法 这里我们使用的一直是一个入口条件循环
</code></pre>
<h3>Count cycle</h3>
<h4>For circulation</h4>
<p>For different types of cycles, of course, we have better writing.</p>
<pre><code class="language-c">for(count=1;count&lt;=max;count++){
	;
}
//这就是更简单的计数循环的写法
//for循环也是自由的 第二部分是任何的关系表达式 第三部分是任何的合法表达式
</code></pre>
<h4>Comma Operators</h4>
<pre><code class="language-c">for(count=1,cost=1;count&lt;=max;count++){
	;
}
//这里的逗号运算符让原本写一个句子的地方能写两个 起到分隔的作用
x = (y = 3,z);
//起到赋值的作用 用的不多 尽量避免
printf(&quot;%d&quot;,a);
//分隔符的作用
</code></pre>
<h3>Exit Conditional Cycle</h3>
<pre><code class="language-c">do{
	printf(&quot;aaaaa&quot;);
	scanf(&quot;%d&quot;,&amp;a);
}while(a!=1);
//此时我们先执行一边函数体 再进行判断 如果满足 继续循环 和前面两种循环规则一样
</code></pre>
<h3>Embedded loops</h3>
<pre><code class="language-c">for(i=0;i&lt;10;i++){
	for(k=0;k&lt;10;k++){
		;
	}
}
//随意嵌套 根据需求
</code></pre>
<h3>Cycle support</h3>
<pre><code class="language-c">for(i=0;i&lt;10;i++){
	for(k=0;k&lt;10;k++){
		continue;
	}
}
for(i=0;i&lt;10;i++){
	for(k=0;k&lt;10;k++){
		break;
	}
}
## continue 结束本次循环 开启该循环中的下一轮循环
## break 循环结束 离开这一层循环结构
</code></pre>
<h3>Array</h3>
<p>The arrays are used to store some relevant new ones of the same type.</p>
<pre><code class="language-c">int array[10];
//建立某个类型的数组
a = array[0];
//访问数组中的元素 从0开始是第一个
//C语言不检查数组的越界 这点需要自行注意
//这被称为可变长度数组 VLA 基本上大部分的编译器都已经支持了它 当然除了宇宙第一IDE
for(i=0;i&lt;10;i++){
	scanf(&quot;%d&quot;,%array[i]);
}
//数组和for循环一起搭配是个很好的注意
</code></pre>
<h3>Use function return value</h3>
<pre><code class="language-c">double power(double n,int p){
	double pow = 1;
	int i;
	for(i=1;i&lt;=+p;i++){
		pow*=n
	}
	return pow;
}
//这就是建立有返回函数的基本方式
//使用函数之前进行声明当然是必要的 不过我们在后面会再次提到
</code></pre>
<h2>Branches and Jumps</h2>
<p>Besides branch and jump, we'll introduce some IO functions and their use here.</p>
<h3>If statement</h3>
<pre><code class="language-c">while(scanf(&quot;%f&quot;) == 1){
	if(tem&lt;FREEZING){
		x++;
	}
	else if(x&gt;0){
		x = 0;
	}
	else{
		tem = 0;
	}
}
#这就是分支也就是if的最基本写法 嵌套的写法也是自然地
ch = gerchar();
## 这个函数不接受任何参数 他只会从输入设备拿一个字符 并且返回
putchar(ch);
## 和getchar相反 这两个函数只用来处理字符 效率比scanf printf 高得多
while((ch = getchar())!= &#39;\n&#39;){
	xxxxxxxx;
}
#这是非常C语言的一个写法 先调用函数 然后把它的结果拿来运算
a &amp;&amp; b
a || b
a !  b
#这三个运算符都是C非常常用的逻辑运算符 我们要避免有些不符合语法的数学语言出现在程序里
#switch case 在这里就不介绍了 和if作用没有差别
</code></pre>
<h3>statement</h3>
<p>Here is a hint that if it's not necessary, we should use it with absolute caution, and that it no longer exists in higher languages, because he doesn't have to cause mistakes.</p>
<pre><code class="language-c">#这个语句需要搭配使用
part2:xxxxxxxx;
goto part2;
#所有的goto使用场景均可以考虑使用其他的语句来代替 使用判断然后goto是最容易引发错误的方法
</code></pre>
<h2>Enter of the character Output Confirm</h2>
