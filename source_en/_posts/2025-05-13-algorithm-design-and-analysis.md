---
title: 'Algorithm Design and Analysis: Divide and Conquer, Dynamic Programming, and Graph Algorithms'
title_zh: 算法设计与分析：分治、动态规划与图算法
date: 2025-05-13 18:32:25 +0800
permalink: /blog/2025/05/13/algorithm-design-and-analysis/
categories:
- Programming
- Computer Science Fundamentals
tags:
- Algorithms
- Dynamic Programming
- Greedy Algorithms
excerpt: Covers divide and conquer, dynamic programming, greedy methods, amortized analysis, graph algorithms, and complexity
  reasoning.
description: Covers divide and conquer, dynamic programming, greedy methods, amortized analysis, graph algorithms, and complexity
  reasoning.
lang: en
translation_key: 2025-05-13-algorithm-design-and-analysis
translation_status: machine
translation_source_hash: 9b4b552c49a57329361b301d6fc262e679a9e0cfef8afdd17dbc28ac2d35dcd3
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Introduction</h2>
<p>How did the algorithm come up? The algorithms are designed to understand the rules, but what other people think of the answer. Why can't we figure out how to handle this.</p>
<p>We need to think of a way to deal with the problem, one that can be universalized in design, one that analyses the problem with one set of ideas, not one that relies on the light.</p>
<p>Basic thinking for scientific research</p>
<ol>
<li>First of all, we'll have a real problem, choosing or finding a real problem.</li>
<li>Next, we want to turn the real problem into an algorithm or a math problem.</li>
<li>And finally, let's deal with this.</li>
</ol>
<p>Basic thinking for dealing with mathematical issues</p>
<ol>
<li>Try to deal with it from the front.</li>
<li>If you can't deal with this problem in front, maybe you can't even think about it.</li>
</ol>
<p>Way1</p>
<ol>
<li>In this case,<strong>Can the simplest case be handled?</strong>？<ol>
<li>When the simplest of things can't be dealt with, just give up.</li>
<li>When the simplest question can be dealt with, deal with him.</li>
</ol>
</li>
<li>When a simple case can be handled, but a complex one can't.<strong>Does decompose work?</strong>What? When it works, then.<strong>Summary method</strong>It's possible.</li>
<li>When decomposition is feasible, and the original problem is one of optimization, and<strong>The best solution to the original problem can be the best combination of small problems.</strong>, so you can take<strong>Dynamic planning</strong></li>
<li>When the conditions are met, and<strong>The best solution can be seen by the local.</strong>♪ Then you can take ♪<strong>Greedy.</strong></li>
</ol>
<p>Way2
When problems can't be decomposed, but need to be studied to solve variations, then other algorithms can be considered, such as linear planning, network flow, etc.</p>
<p>Way3
When it's not possible, it's necessary to study the enumerator, so we can think about a better enumeration strategy.</p>
<p>Last Way
And finally, there's no simir question, try simir algorithms.</p>
<p>The design of algorithms in the AI era, in which there are many randomly tried methods, may be solidified by the use of neural networks and large amounts of pre-collected data, which sometimes increase the efficiency of computing. That's...<strong>Use NN to replace a very complex module</strong></p>
<h2>Divide Conquer</h2>
<p>Basic thinking: Many of the problems in the real world are regressive, and the only difference between big and small is size. To solve big problems, we can break them down into small ones and deal with small ones through retrogression, so we can combine the solution of the original problems.</p>
<ol>
<li>Divide solves small problems</li>
<li>Conquer handles minor issues (procedural regression)</li>
<li>Combine got a big problem.</li>
</ol>
<p>The core of Divide and Combine is how to observe the form of the data structure for output and output, and he determines how we divide and merge, and we might look at how they divide and merge, arrays, arrays, pools, trees, maps, most commonly used data structures.</p>
<p>For an array, each element has two attributes: value and subscript;</p>
<p>So it's easy to get a basic subdivision based on the subscripts, taking one of the elements out, divided into two arrays.&#36;n&#36;individual&#36;n-1&#36;An element. Then, for the larger group, the split is repeated until the simplest form of the two elements that we will be dealing with is finally recombined. Corresponds to sorting tasks, which is inserting sorting.</p>
<p>In another way, it divides from the middle into essentially large sequences and repeats them. The final merger should start in two groups and select the smallest of the elements. And the two groups that have been sorted, the smallest elements must be on the left-hand side, starting with the left-hand side, and each time they can remove the smallest elements, reducing the consumption of the sort. It's sorted.</p>
<p>One way to divide is to consider dividing from the values of the arrays; randomly choosing a pivot dollar, with elements larger than it as group A, with elements smaller than him as group B, and then retrieving the call sorting functions of the two groups separately until all the elements are sorted correctly, and now there is no need for cobine, a sorting sequence that can be regrouped, which is quick.</p>
<p>There's a lot of questions to use if we want to find out.&#36;A&#36;Number of elements&#36;k&#36;Large, we can use a similar fast-paced method, by value. Select a pivot randomly, using the larger elements as group A, the smaller elements as group B, and then write a logic to determine which group the larger element of k is in and then revert to the function. Finally back to the element found.</p>
<p>For graphic data, which are also more common in algorithms, we can consider achieving diviide by reducing the vertex and side, and starting with the simplest case.</p>
<p>It's natural for divide to be a matrix, and we'd like to split the matrix according to location until it becomes smaller. This is used in high-speed matrix multiplication (or more matrix-related issues). In fact, the big multiplication is using the same divide approach.</p>
<p>To speed up diviide conquer, the two core lines are:</p>
<ol>
<li>Fewer fractions that need to be calculated directly, i.e., new fractions are processed using the addition of the fraction</li>
<li>Smaller fractions of one-time fractions, smaller fractions of time. degrees</li>
</ol>
<p>If an issue that can divide is a matter of optimization and can be synthesized by sub-issues, we will have to consider using dynamic planning, which is a further reflection of divide conquer. The greed behind it would also be an improvement in dynamic planning.</p>
<h2>Dynamic Planning</h2>
<h3>DP Foundation</h3>
<p>The issue of dynamic planning continues to be one that needs to be divided, except that here we would like to present more ideas for turning issues into sub-issues, which this section hopes to learn — the way to think and deal with issues using DP thinking.</p>
<p>If our core problem is finding optimal decision-making, and the problem can be modelled as a multi-step decision-making process, then we can consider using DP.</p>
<p>Let us begin by introducing a basic example: there are three aircraft, 10 missiles, each with its own probability of shooting down the number of missiles and its own value, and how best to achieve the highest overall value of shooting down.</p>
<p>It's not realistic for larger sizes, but for this problem involving pure integers, we can consider using a different type of diviide that uses multi-step decision-making methods (because pure integers mean that options for decision-making are limited).</p>
<p>Definitions: Functions&#36;max{V(x,y)}&#36; For...&#36;y&#36;A missile. Destroy.&#36;x&#36;Maximum value an aircraft can get</p>
<p>So at this point, we want to calculate&#36;max{V(3,10)}&#36; We're actually making decisions about how many missiles each aircraft needs to use, so we can start with the first decision, directly listing the first one.&#36;maxV(3,10)=max{max(2,x)+EV(x)}&#36; Yes.&#36;10-x&#36;The desired value of one aircraft attacking the first aircraft and the other being transferred from another.&#36;maxV&#36;function.</p>
<p>By using such a computational procedure, it would be possible to simplify the problem to the case of an aircraft and to finalize the resolution. Because the simplest form of this model, an aircraft and any number of missiles are easy to solve.</p>
<p>The idea of multistep decision-making and optimizing total value is<strong>Bellman equation</strong>Such a solution would be feasible if the problem could be summed up as a multi-step decision-making process and there was a regressive relationship between optimal resolution.</p>
<p>The idea of a multi-step decision-making process is understood.<strong>Modelling issues related to decision-making into multi-step decision-making structures</strong>One step at a time to optimize the solution to complex problems.<strong>The choice of each step of decision-making should be simple in itself and should not have too many branches</strong>, after decision-making results and further back-to-back decision-making should be made as easily as possible by code and not generate a large number of branches of f</p>
<p>The multi-step decision-making described above is in fact a step-by-step approach, with multiple layers of back-to-back and all calculations producing many performance costs, and we need to consider ways to reduce multi-step decision-making costs. dp The result of storing a sub-issue when it is actually used, and when the sub-issue needs to be solved again, can be directly identified to reduce the eventual complexity.</p>
<h3>3 classic DPs</h3>
<p>Minimum number of array operations: a series of arrays of product, total&#36;A_1,...,A_n&#36;Total&#36;n&#36;A matrix, each size&#36;p_0,...p_n&#36; The minimum total number of operations can be obtained in the order of operation, note:&#36;p_0,p_1;p_1,p_2&#36;The total number of arrays multiplied is &#36;p_0p_1p_2&#36; </p>
<p>Such an issue can naturally be transformed into a situation of multi-step decision-making, and we need to decide which matrix to multiply first. And how to turn this long matrix product sequence into a short matrix with similar problems requires only the addition of brackets and a break point for the final product.</p>
<p>Definitions:&#36;OPT(i,j)&#36; Yes.&#36;i,j&#36;Minimum number of operations for the upper matrix product (for this type of string, even if it is a single string, but is still used to tracking in two positions, e.g. number of returns, valid brackets length), so an iterative formula can naturally be given
&#36;&#36;OPT(i,j)=min{OPT(i,k)+OPT(k+1,j)+p_{i-1}p_kp_j}&#36;&#36;
Use this type of regression formula to reduce the length of the longest series by two matrix multipliers.</p>
<p>String Matching: Give two strings <code>OCCURCE,OCUORCE</code> The match between the two strings is judged by adding one point when the matching position is perfect, and three points when the matching error or omission is subtracted to calculate the largest matching score.</p>
<p>It's two long string sequences, too long string that we can't handle, so we want to reduce the length. The simple idea is to start at the end, and so choose.&#36;OPT(i,j)&#36;For string 1 to&#36;i&#36;Zen, two-by-two.&#36;j&#36;location, the degree to which two strings match.</p>
<p>There are three possible matches for the last character of the two strings: &#36;E-E;E-NULL;NULL-E&#36; Then you can give it to me.&#36;OPT&#36;One form of decision-making is
&#36;OPT(i,j)=max~begin{aligned}
&amp;s(i,j)+OPT(i-1,j-1)\
&amp;s(i,NULL)+OPT(i-1,j)\
&amp;s(NULL,j)+OPT(i,j-1)\end{aligned} &#36;
This is actually how to determine the match of the last element, and then turn the long string into more strings, thereby making the OPT defined here iterative.</p>
<p>Maximum common subseries: given two strings <code>text1</code> and <code>text2</code>returns the longest of these two strings <strong>Common Subseries</strong> The length. If not present <strong>Common Subseries</strong> Other Organiser <code>0</code> I don't know. A string. <strong>Subseries</strong> refers to a new string: it is a new string formed by the original string without changing the relative order of the characters.</p>
<p>We still have two strings that need to be tracked, too long strings that are difficult to process, so we still want to return to reduce the length of the string, or to track the string from the tail.&#36;OPT(i,j)&#36;String 1 &#36;i&#36;Previous with string 2 &#36;j&#36;The longest public sub-series before that can be given in an iterative manner by determining the match of the last position.
&#36;OPT(i,j)=\begin{aligned}
&amp;0<del>if</del> i=j\
&amp;OPT(i-1,j-1)+1 <del>if</del>T[i]=T[j]\
&amp;max{OPT(i-1,j),OPT(i,j-1)<del>if</del>T[i]\ne T[j]}\end{aligned}&#36;&#36;</p>
<p>The idea of a back-to-back issue in dp does not require that we have to re-entry to use the function that produces the question, and we define ourselves a new function that produces some transitional results.</p>
<p>This OPT function is defined in order to simplify the problem into the simplest form (which we can deal with directly) by using them in a step-by-step decision-making that reduces the complexity of the problem.</p>
<h3>Define new OPT function</h3>
<p>Maximum reply: Give you a string s, find the longest entry in s. "bab" is a 3-long substring reply.</p>
<p>Naturally, for the length of 1 we know that he must have satisfied his reply and that we should not create an OPT function to reduce the length (which must not be a reduction from the head or tail, but rather a symmetrical reduction).&#36;OPT(i,j)&#36;This means the longest reply string in this section, which is used to determine whether or not the string in this section is returned, so you can give the attribute
&#36;&#36;OPT(i,j)= OPT(i+1,j-1) ~~if ~~s[i]==s[j] &#36;&#36;
When?&#36;j-i==1&#36;Or...&#36;j-i==2&#36;When you don't think about the strings.</p>
<p>Maximum valid brackets: A string containing only `(' and `) ' finds the length of the maximum valid (in the correct format and continuous) string.</p>
<p>The maximum valid parenthesis length of a single-string tracking study ending with i could be considered. This would make it possible to establish a regression; it could also consider the use of two indicators to track the validity of the brackets in the study compartment, which could be created by removing the last brackets (the last brackets would have to be)&#39;)&#39;It's legal.</p>
<h3>DP on array</h3>
<p>Many of the questions that deal with arrays, they do not have visible decision-making, but the simplest questions we would like to discuss are short arrays, while the real problem is long arrays, and we are actually building a regression equation (bellman equation) to shorten the length of arrays. Creates a new opt function, where the solution of a real problem is hidden in the opt function, and then studies how to deal with this opt function in a reverse manner.</p>
<p>The function of the opt that can be created should be diverse, and it is important to change the idea of designing the opt function in a timely manner when dealing with difficulties. The probability of multi-step decision-making is not the only one, so when dp is used, it is also necessary to be flexible in thinking and to use different methods to minimize big problems.</p>
<p>The purpose of decision-making is to simplify complex issues, and because the natural length of the array itself determines the complexity of the problem, the core of this dynamic planning lies in finding the sub-problems of shorter and less complex problems, and in finding the relegation of problems between less and less complex. Finds a retrieving relationship between the increasingly short opt function. The reduction of arrays is simple, except that the length is reduced before/from.</p>
<p>In sum, it is not scary not to find a decision when dealing with arrays, and it is often easier to deal with arrays, with more fixed and rigid thinking.</p>
<p>Maximum upscaling subseries: Give you an integer array of Nums to find the length of the maximum strict increment sequence. Subsequences are sequences derived from arrays, removing (or not removing) elements from arrays without changing the order of the remaining elements.</p>
<p>This issue is well dealt with for an array of only one or two elements, but it has not grown so well, so we need to consider using the DP approach to reduce the length of the array. For this single-string array, the best method of decision-making is to create an OPT function with a reduced length from the end of the array.</p>
<p>So define&#36;OPT(i)&#36;For the right time.&#36;i&#36;An element, the maximum ascending subseries of an array. I'll give you an extension.
&#36;OPT(n)=OPT(n-1)+1 <del>if</del> num[n]&gt;num[n-1]&#36;&#36;
<strong>It's used here to fit the first element, not the first one, so it's easier to give this line of regression.</strong>, and we've made this self-definition of the use of strings before, and finally all of them.&#36;OPT[i]&#36;Just one max.</p>
<p>Maximum subsequences: set an integer array Nums, find a continuous sub-set with the maximum sum of one element.</p>
<p>It is also the problem of arrays, which naturally are easy to handle and complicated, and we want to reduce the length of arrays to make them simple. Definitions&#36;OPT(i)&#36;It's the largest number of consecutive sub-sets at the end of the number i, naturally giving in.
&#36;&#36;OPT(i) = max(OPT(i-1)+num[i],num[i])&#36;&#36;
As with the last small question, what we've created is the exact end of the number I, because it's easier to construct and process it.</p>
<h3>DPs that may be in circulation</h3>
<p>There is a sequence of a sequence to the ringless; the shortest road question above him can be understood as a multistep decision-making process in reverse and sequence.</p>
<p>When there is a ring on it, the shortest-path problem is circular dependence, and the return of a dead cycle is called, which must be addressed. The most common method at this point is<strong>Adds a parameter for a recursive function to keep him down while he returns. It's actually increasing the particle size of the decomposition structure to make the problem more detailed.</strong>I don't know. In fact, this idea of making matters more detailed is the same as the one that preceded the construction of the new opt. The right PT is the core.</p>
<h2>Greedy</h2>
<h3>Greedy Thought</h3>
<p>In dynamic planning, we don't know what the best decision is at this time when we are somewhere, so we need to count back to the full to know the final result. And sometimes we know where the best decisions are, so we can introduce Greedy.</p>
<p>Selection issues: giving&#36;N&#36;Courses, each with the corresponding time and number of participants&#36;W_i&#36;How the curriculum is scheduled to be supplemented and maximized</p>
<p>The big questions are difficult to solve, but the sub-structure (only 1-2 lessons) is easy to judge by a few ifs and an optimisation problem, and dynamic planning is likely to address this problem well. The problem of scheduling can also be seen in some way as an array. Shorter length is at the heart of the solution, so we can start from the end of the class backwards so that we can trace less information and define it.&#36;OPT(S,T)&#36;Yes.&#36;T&#36;Time is available. There's still time.&#36;S&#36;The maximum number of scheduled classes can be cancelled in reverse.
&#36;&#36;OPT(S,T)=max{OPT(S-课x-冲突课,课x的上课时间)+W_x,OPT(S-课x,T) }&#36;&#36;
The last class of multi-step decision-making does not lead to the final solution of the big problem, and the reduction from the last class is just to reduce design.&#36;OPT&#36;The difficulty.</p>
<p>♪ When all&#36;W_i&#36;In the first place, we can use Greedy to deal with this problem, and we can find the best solution without having to go through all the scenarios by constantly choosing the first class/last-night course and synchronized conflict-free courses.<strong>We don't need to solve the problem, we choose the best part.</strong></p>
<p><strong>Dynamic planning and Greedy are very similar, they're mainly used to handle optimization and are based on sub-structure analysis, and all Greedy has a corresponding DP behind him, listing all situations instead of choosing local merits, and they can deal with such issues.</strong></p>
<p>Or is there a question of what kind of decision should we make with Greedy? The selection of the right option by its own Idea is wrong, for example, by scheduling the earliest course/show the shortest course, and finding the greed strategy ahead is not an easy task.</p>
<p><strong>You can't rely entirely on the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the rules of the law of the law of the law of the world, but of the rules of the rules of the rules of the law of the world, of the rules of the rules of the law of the world, of the rules of the world, of the people of the world, of the world, of the people of the world, of the world, of the world, of the world, of the world, of the world, of the world, of the world, but of the world, on the rules of the law, of the rules of the law of the world, of the world, of the world, of the world, and the world</strong></p>
<h3>Greedy's theory.</h3>
<p>The Greedy method is used to deal with the following problems, and to set up a collection.&#36;S&#36;Find a subset of him.&#36;A&#36;Jean.&#36;F(A)&#36;Max. In the absence of a greedy approach, we can only go through the whole assembly to find all possible subsets. Of course, Greedy can only deal with two of these kinds of problems.</p>
<ol>
<li>&#36;f(S)=\sum w_i&#36; That means that each sub-component has nothing to do with the whole function.</li>
<li>&#36;f(x)&#36;It's not linear, but he's condensed.</li>
</ol>
<p>Question 1: Finding a very irrelevant group of vectors, at which time each vector has its own value, and we want to maximize the value of the last very irrelevant group.</p>
<p>This is naturally summarised as multi-step decision-making, using the DP algorithm to deal with the issue, which can be addressed through iterative end-to-end decision-making on whether or not to add the last vector (similar to the scheduling issue).</p>
<p>But at this point, we want to maximize value functions to be linear for each subset, so we can consider the greed strategy and keep introducing the highest value vector until we do not satisfy the unrelated group.</p>
<p>The Prim algorithm, Kruskal algorithm, and the Dijkstra algorithm also use greedy tactics, because their total distance is also a linear function, and the search for the smallest-generated tree and the shortest path are a subset, so the Greedy strategy is feasible and the next time a similar problem is encountered, it is possible to consider Greedy instead of the DP that uses multi-step decision-making.</p>
<p>We also discussed a situation where Greedy could be considered, since it was a pool function at this point, and when he satisfied the nature below, he could consider Greedy.
&#36;&#36;f(A)+f(B)\ge f(A\cup B)+f(A\cap B)&#36;&#36;
Definition of Equivalence
&#36;&#36;&#36;if<del>S_{1}\subset S_{2} ~then</del>f(S  e)-f (S   )\gef (S 2+e)-f (S 2) &#36;
This is a near-uniform, and at least the subset extension function will not be reduced.</p>
<p><strong>This definition reflects a quasi-combust nature, where the addition of elements increases the overall function faster when the subset is smaller, and when the subset is larger, the increase in the function is less obvious, and the increase is related to the base.</strong></p>
<p>Why does this thing work? Because he can provide a higher line,&#36;S\subset T,T-S=E&#36; And...&#36;e_i&#36;Yes.&#36;E&#36;All elements of the list are available
&#36;&#36;f(T)\le f(S)+\sum e_i&#36;&#36;</p>
<p>What kind of greedy strategy should be followed when confronted with such functions? Find the largest addition, the largest marginal increase. So that the increase will be smaller and smaller, and the solution that we finally find will be near the best. If similar problems are encountered (e.g., backpack problems), a near-negative strategy may be considered to find the most expensive elements.<strong>Find a greedy indicator and then implement greed.</strong></p>

<p>For context, compare <a href="/en/blog/2025/05/12/data-structures-introduction/">the data structures introduction</a> and <a href="/en/blog/2024/12/18/search-and-sorting-algorithms/">the search and sorting algorithms notes</a>.</p>
