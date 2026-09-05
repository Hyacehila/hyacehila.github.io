---
title: Web Frontend Basics Overview
title_zh: Web 前端基础概述
date: 2025-05-15 17:16:45 +0800
categories:
- Programming
- Full Stack Development
tags:
- HTML
- CSS
author: Hyacehila
mathjax: false
hidden: true
excerpt: Covers HTML structure, common elements, links, images, tables, forms, meta information, audio and video, CSS colors,
  text, and the box model.
description: Covers HTML structure, common elements, links, images, tables, forms, meta information, audio and video, CSS
  colors, text, and the box model.
excerpt_zh: 整理 HTML 页面结构、常见元素、链接、图像、表格、表单、meta 信息、音视频、CSS 颜色、文本和盒模型。
permalink: /blog/2025/05/15/web-frontend-basics-overview/
lang: en
translation_key: 2025-05-15-web-frontend-basics-overview
translation_status: machine
translation_source_hash: 47f4452b74fadc546a78177b48005ce271e3256acc41786d85a3e5288283fb0f
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>HTML Profile</h2>
<p>HTML is a language used to describe a web page, fully known as Hyper-Text Markup Language, or hypertext tagging language. The text, buttons, pictures, videos, etc. that we see when we view the web page are presented through HTML writing and through browsers. This will be at the heart of our presentation at the front end.</p>
<p>Read this article alongside <a href="/en/blog/2026/06/08/fact-layer-interface-layer-markdown-html/">Factual and Interface Level: Markdown and HTML are not a substitute</a> to compare how related concepts develop in different contexts.</p>
<p>In October 2014, HTML5 was issued as the recommended criterion for stabilizing W3C, which means that the standardization of HTML5 has been completed. HTML is already a very mature language, and he is the basis for follow-up throughout the front end.</p>
<p>All computers with browsers are able to view HTML files, using browsers only to modify the suffix, or using some plugins to modify and preview HTML codes in VSC.</p>
<h2>HTML Foundation</h2>
<h3>HTML Infrastructure</h3>
<p>The HTML code consists of characters in sharp brackets called HTML elements, each of which includes an initial and an end label with an end label with an additional slash. Each HTML element conveys between labels to the browser. The label provides the browser with a structure to render content</p>
<pre><code class="language-html">&lt;!DOCTYPE html&gt;
&lt;html lang=&quot;zh-CN&quot;&gt;
&lt;head&gt;
    &lt;!-- 声明文档使用的字符编码为 UTF-8，避免中文乱码 --&gt;
    &lt;meta charset=&quot;UTF-8&quot;&gt;
    &lt;!-- 设置页面在移动设备上的显示效果 --&gt;
    &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width, initial-scale=1.0&quot;&gt;
    &lt;!-- 定义网页的标题，会显示在浏览器标签页上 --&gt;
    &lt;title&gt;我的第一个 HTML 页面&lt;/title&gt;
&lt;/head&gt;
&lt;body&gt;
    &lt;!-- 一级标题 --&gt;
    &lt;h1&gt;欢迎来到我的网页&lt;/h1&gt;
    &lt;!-- 段落 --&gt;
    &lt;p&gt;这是一个简单的 HTML 示例，用于帮助新手学习 HTML 基础。&lt;/p&gt;
    &lt;!-- 无序列表 --&gt;
    &lt;ul&gt;
        &lt;li&gt;HTML 是超文本标记语言&lt;/li&gt;
        &lt;li&gt;用于创建网页的结构&lt;/li&gt;
        &lt;li&gt;由各种标签组成&lt;/li&gt;
    &lt;/ul&gt;
    &lt;!-- 插入图片，需替换为实际图片路径或 URL --&gt;
    &lt;img src=&quot;example.jpg&quot; alt=&quot;示例图片&quot; width=&quot;300&quot;&gt;
    &lt;!-- 创建超链接 --&gt;
    &lt;a href=&quot;https://www.example.com&quot;&gt;访问示例网站&lt;/a&gt;
    &lt;!-- 水平线 --&gt;
    &lt;hr&gt;
    &lt;!-- 版权信息 --&gt;
    &lt;p&gt;&amp;copy; 2025 我的网页版权所有&lt;/p&gt;
&lt;/body&gt;
&lt;/html&gt;
</code></pre>
<p>From this section, the HTML infrastructure can be understood.<code>&lt;html&gt;</code> Elements state that all of them are HTML codes.<code>&lt;body&gt;</code> Declare that they are what needs to be presented in the browser main window.<code>&lt;h1&gt; &lt;p&gt;</code> Declares that they are first-level titles or paragraphs.</p>
<p><code>&lt;html lang=&quot;zh-CN&quot;&gt;</code> Wait for a space to be added to the HTML label and then follow a set of assigned values to add features to the label to provide additional information for the label. The particular language here limits the subsequent language. Features are predefined, and we need to know what the labels are for.</p>
<p><code>&lt;head&gt;</code> To declare page information, not in the main window, e. g.<code>&lt;title&gt;</code>If you want to declare a page title, it will be displayed above the URL address or at a small tab.</p>
<p>As for the other elements, we will gradually describe in the following narrative that the role of all common elements needs to be understood.</p>
<h3>Base Elements</h3>
<h4>Structure Information</h4>
<p>Space: To enhance code readability options for developers, a large number of lines and spaces are treated as one.
<code>&lt;h1&gt;--&lt;/h1&gt;</code> Title control component, six.
<code>&lt;p&gt;--&lt;/p&gt;</code>  Paragraph in text
<code>&lt;b&gt;,&lt;i&gt;</code> Bold and italics
<code>&lt;sup&gt;,&lt;sub&gt;</code> Superscript & Subscript
<code>&lt;br /&gt;,&lt;hr /&gt;</code> Line-repeated symbols and horizontal lines, which are usually used individually as an expression of meaning</p>
<h4>Semantic Information</h4>
<p>Semantic information titles do not change the structure of the web page, but add additional information, some of which is briefly presented here</p>
<p><code>&lt;strong&gt;,&lt;em&gt;</code> To emphasize that, by default, in bold and in italics
<code>&lt;blackquote cite = &quot;xxx&quot;&gt;</code> Quote to a hyperlink, click on the contents of the block to jump
<code>&lt;abbr title = &quot;xxx&quot;&gt;</code> Provide abbreviations, which are in the form of abbreviations,<code>title</code>It's a full spell, and a floating mouse will provide a full spell.
<code>&lt;cite&gt;</code> Internal information is quoted
<code>&lt;dfn&gt;</code> Internal information is new.
<code>&lt;address&gt;</code> Provide contact information
<code>&lt;ins&gt;,&lt;del&gt;,&lt;s&gt;</code> Underlined, strikeout, strikeout</p>
<h4>List</h4>
<p>HTML5 provides lists of three basic forms
<code>&lt;ol&gt;</code> to create an orderly list, using <code>&lt;li&gt;,&lt;/li&gt;</code> Create list items
<code>&lt;ul&gt;</code> to create an unsequenced table, which uses <code>&lt;li&gt;,&lt;/li&gt;</code> Create list items
<code>&lt;dl&gt;</code> For the creation of the list of definitions, he included a range of terms that needed to be defined and the multiple definitions that were needed for that term. uses <code>&lt;dt&gt;,&lt;/dt&gt;</code> Creates a defined term for the list,<code>&lt;dd&gt;,&lt;/dd&gt;</code> Used to explain its definition. As a term may have multiple definitions, the same definition may be consistent with multiple terms, so the HTML code between them does not contain a relationship.</p>
<p>List allows nesting, in<code>&lt;li&gt;,&lt;/li&gt;</code>to create a new list</p>
<h3>Links, images, tables</h3>
<h4>Link</h4>
<p>Links are the main specificity of the network, and we allow jumps between multiple pages, which is the core of web browsing. The links consist of a variety of different types of features, and a series of specifications for achieving the links have been agreed between HTML and other external software, allowing us to easily prepare the relevant codes.</p>
<p>Basic links used <code>&lt;a hred = &quot;xxxx&quot;&gt; text &lt;/a&gt;</code> Composition, including linked web pages and descriptive texts. Link needs to contain a complete URL (web site) to unspecify a specific page.</p>
<p>When we need to point to the other pages of the same website, we don't need to use a long-domain name, we can use a short URL, we just need to use the relative directory structure to introduce the other ones.<code>.html</code> The end page is fine and the parent page is available. <code>..</code> Achieved.</p>
<p>Use the following structure if you want to use Email<code>&lt;a hred = &quot;mailto:hyacehila@outlook.com&quot;&gt; Email Hy &lt;/a&gt;</code></p>
<p>If you need to automatically open in a new window, you need to set features <code>&lt;a hred = &quot;xxxx&quot; target=&quot;_blank&quot;&gt; text &lt;/a&gt;</code></p>
<p>If you need to link to the specific location of the current page, first we need to set specific settings in other elements of this page:<code>&lt;h2 id = &quot;feature&quot;&gt;</code>  At this point, replace the connection with <code>&lt;a href = &quot;#feature&quot;&gt;</code> This is going to be an internal page jump</p>
<p>We also need to know what he set up if a link is needed to a specific location for another page (whether the page is external or inside the site) <code>id</code> Then use<code>&lt;a hred = &quot;xxxx/#feature&quot;&gt; text &lt;/a&gt;</code></p>
<h4>Image</h4>
<p>Pictures on the website need to be stored as attachments before reference is made, and it is therefore recommended that one or more files be stored separately in the documents of the website <code>images</code></p>
<p>If you want a picture on the website, you need to use the empty elements below (no end label elements) <code>&lt;img src = &quot;images/quokka.jpg&quot; alt = &quot;xxxx&quot; title = &quot;xxxxx&quot;&gt;</code> They show the relative position of the picture, the alternative text description (for barrier-free design) when the picture cannot be viewed, and the title of the picture.</p>
<p>If we need to specify the size of the image manually, use specific <code>&lt;height = &quot;450&quot;,width = &quot;430&quot;&gt;</code> But now this page layout usually uses the CSS file to specify, including matching settings</p>
<p>Image blocks can be embedded in other blocks, which is at the heart of the various layout methods that HTML uses. The effect of placing him within or outside a block element is completely different.</p>
<p>New version of HTML5 used <code>&lt;figure&gt;</code> Block instead of original empty elements <code>&lt;img&gt;</code> The basic method of use is <code>&lt;figure&gt; &lt;img src = &quot;images/quokka.jpg&quot; alt = &quot;xxxx&quot;&gt; &lt;figurecaption&gt; xxx &lt;/figurecaption&gt; &lt;/figure&gt;</code>
Supported adding instructions for pictures</p>
<p>HTML currently supports a wide range of pictures in a variety of formats, including PNG, JPG, GIF, SVG, etc.</p>
<h4>Table</h4>
<p>The table clearly explains the complex information. Here we look at how to construct a table. HTML prepares a table by line.</p>
<p><code>&lt;table&gt;</code> To start a table block <code>&lt;tr&gt;</code> Each line used to start the table <code>&lt;td&gt;</code> To start every cell <code>&lt;th&gt;</code> and <code>&lt;td&gt;</code> It's the same thing, but it's actually meant to mean the title.<code>&lt;th scope = &quot;row&quot;&gt;,&lt;th scope = &quot;col&quot;&gt;</code> Specifies whether this is a row or column title</p>
<p>Yes. <code>&lt;th&gt;</code> and <code>&lt;td&gt;</code>  Add features <code>&lt;th colspan = &quot;2&quot;&gt;</code> It allows him to cross multiple columns, the same characteristics. <code>&lt;th rowspan = &quot;2&quot;&gt;</code> We can get him across a lot of lines.</p>
<p>Use <code>&lt;thead&gt;,&lt;tbody&gt;,&lt;tfoot&gt;</code> The tables can be divided into multiple components that help readers understand that the control methods for each cell remain unchanged.</p>
<p>The HTML code only controls content, and as for the display style, it is controlled by CSS.</p>
<h4>Form</h4>
<p>The form is a place that provides some blank areas for you to fill in, and can be used when we need to collect information from visitors. Many browsers collect user information in advance and automatically fill in according to the form requirements.</p>
<p>The form is filled in by the user, then the information is submitted to the server by clicking the button, and the server returns a new web page accordingly.</p>
<p>Use a form <code>&lt;form action = “URL” methon = &quot;get&quot;&gt; &lt;/form&gt;</code> Create among them <code>action</code> I've decided where the server will take the URL.<code>method</code> Method of submission determined, generally small search used <code>get</code> ; upload files with password confirmed <code>post</code></p>
<p>Add one to the form <code>&lt;input type = &quot;text&quot; name = &quot;username&quot; maxlength = 4&gt;</code> Can create a text box, of which <code>name</code> It's a key that matches the server.<code>maxlength</code> is the maximum length allowed;replacement<code>text</code>Yes<code>password</code>, and then automatically hide the entered character</p>
<p>If you want to create a multiline text box, you should use <code>&lt;textarea name = &quot;xxx&quot;&gt; &lt;/textarea&gt;</code> Achieved</p>
<p>Select button to use <code>&lt;input type = &quot;ratio&quot; name = &quot;xxx&quot; value = &quot;xxx&quot;&gt;</code> We need to be in one.<code>&lt;form&gt;</code> Use multiple single buttons to achieve single theme</p>
<p>Recheck button to use <code>&lt;input type = &quot;checkbox&quot; name = &quot;xxx&quot; value = &quot;xxx&quot;&gt;</code> We need more than one check button to achieve it.</p>
<p>Click to use <code>&lt;select name =&quot;xx&quot;&gt; &lt;/select&gt;</code> with multiple <code>&lt;option value = &quot;xxx&quot;&gt; &lt;/option&gt;</code> Composition</p>
<p>Use if users want to upload files<code>&lt;input type = &quot;file&quot;&gt;</code> Achieved; correspondingly, if an additional submission button is required <code>&lt;input type = &quot;submit&quot;&gt;</code></p>
<p>HTML5 provides more common form properties including email url search, etc. <code>&lt;input&gt;</code> Selection is sufficient to facilitate user completion of forms.</p>
<h3>Other Tags</h3>
<h4>Version Information</h4>
<p>At the beginning of the entire page, we can use <code>&lt;!DOCTYPE html&gt;</code> Statement Version</p>
<p>In HTML code, we can use <code>&lt;!-- XXXX --&gt;</code> Other Organiser</p>
<p>Each HTML component is usable <code>&lt;id = &quot;xxx&quot;&gt;</code> This will facilitate the automated processing of the back JS.</p>
<p>Those elements that are always in a different line are called block elements. <code>&lt;h1&gt; &lt;p&gt; &lt;ul&gt; &lt;li&gt;</code> etc; conversely called the inner chain element</p>
<p><code>&lt;div&gt; &lt;/div&gt;</code> The elements allow us to concentrate a group of elements in a single block for easy management, and when the web page distinguishes a lot of important pieces, we often use them.</p>
<p><code>&lt;iframe&gt;</code> Embedded a new page in a small window of the page</p>
<h4>Page Meta Info</h4>
<p><code>&lt;meta&gt;</code> Put it on. <code>&lt;head&gt;</code> ,use <code>&lt;meta name = &quot;xxx&quot; content = &quot;xxx&quot; /&gt;</code> Composition, which can include the designer of the website <code>&lt;author&gt;</code> Page Cache <code>&lt;pragma&gt;</code> With Expiry <code>&lt;expire&gt;</code> etc.;</p>
<p><code>&lt;descirption&gt;</code> Introduction to Page Description <code>&lt;kerwords&gt;</code> I can give you key words.<code>&lt;robots&gt;</code> You can give permission to reptiles, etc.</p>
<h4>Audio and Video</h4>
<p>Video and audio are also very important on the modern Internet, which has led to significant improvements in HTML 5.</p>
<p>Use <code>&lt;video src = &quot;xxx.mp4&quot;&gt; &lt;/video&gt;</code> You can create a video block to give the address of the video. There are also a lot of specific options, such as playout and cover, to adjust when needed.</p>
<p>Corresponding. If you need audio, just...<code>&lt;audio src = &quot;xxx.mp3&quot;&gt; &lt;/video&gt;</code> Yeah.</p>
<h2>CSS Profile</h2>
<h3>Basic introduction</h3>
<p>CSS is used to create rules that specify how the content of the elements is displayed; once the CSS rationale is presented, we need only know the various styles.</p>
<p>To understand CSS, we need to know that every HTML element is in a box. We can set the element format rules for the contents in this box. The CSS's role is to select the cells and then give the rules.</p>
<p>A CSS rule consists of two parts, a selector and a rule. Like <code>p,h1 {font-family:Arial,...}</code>  First Part Selector<code>p,h1</code>block, the latter part is separated by a colon.</p>
<p>In practical applications, CSS documents are generally written separately, at which point they can be <code>&lt;head&gt;</code> Use empty elements <code>&lt;link&gt;</code> Sets the link position for the CSS file.</p>
<p>CSS has a wide variety of choices that allow us to apply the rules to a wide range of locations, including, inter alia, universal selection, type selection, class selection, ID selection. When different CSS rules are set for a selection at the same time, then the choicer is better and more specific.</p>
<h3>Colour</h3>
<p>CSS uses code <code>&lt;color:&gt;</code> Specifies the colour, using RGB values, 16-digit encoding, and preset names.</p>
<p>Besides the color of the elements, we can set the background colour. <code>&lt;background-color&gt;</code>  Whether it's color or background, we can do it alone. <code>&lt;opacity:&gt;</code> Set Transparency</p>
<p>After CSS3, we can use HSL and HSLA colours, similar to RGB, using three values to control colours such as: <code>&lt;color:hsl(0,0%,78%)&gt;</code> Separately controlling the tone, saturation, visibility and one additional control transparency.</p>
<h3>Text</h3>
<p>The control text is divided into two broad categories, the CSS rules that directly affect font and appearance properties, and rules that do not concern text fonts.</p>
<p>We can use it. <code>font-family: </code> Set fonts so that, in order to avoid users missing fonts, we can specify a series of fonts, such as: <code>font-family: xxx,xxx,xxx; </code> Let computers be selected according to the circumstances.</p>
<p>To specify the size of the font, we need to use <code>font-size:</code> The rules for permitting use include: <code>px,%,em</code> is the pixel size statement, the percentage statement relative to the standard size, and according to the width of the font m.</p>
<p><code>&lt;font-weight:bold&gt;</code> Set bold;<code>font-style:italic</code> Set italics;<code>&lt;text-transform:&gt;</code> Set case;<code>&lt;text-decoration:&gt;</code> (b) Setting up decoration lines;</p>
<p>Control spacing, alignment, indentation code omitted here. We can discuss it when we need to use it, and we need only basic understanding here.</p>
<h3>Box.</h3>
<p>To control the size of the box, we can use it. <code>height:xx;width:xx;</code> This box can be a block element or use. <code>&lt;div&gt; &lt;/div&gt;</code> Defines the block.</p>
<p>In order to fit the user's screen, many HTML designs allow automatic scaling, which we can use to oversize or narrow. <code>min-width:450px;max-width;650px;</code></p>
<p>Similarly, we can set the maximum and minimum height to control the size of the user screen.  <code>min-height:450px;max-height;650px;</code> We can use it when there's too much over there. <code>overflow</code> Parameters, set to <code>hidden/scroll</code> Hides the spill or allows the user to slide.</p>
<p>Use <code>border-width;border-style;border-color</code> You can set the size, form and colour of the box border. Use <code>padding:10px;</code> to set the inner distance of the border; to use <code>margin:10px;</code> You can set the distance outside the border.</p>
<p><code>display</code> (a) the way the element is displayed can be controlled (the element can be hidden directly and can only be discovered by looking at the source code); <code>visibility</code> The display of the box can be controlled (preserve the space occupied by the element but leave it blank)</p>
<h3>Lists and Tables</h3>
<p><code>list-style-type</code> Allows us to control the style of bullet in the list</p>
<p><code>list-style-position</code> Allows us to control the position of item symbols in the list, whether to mix them into the text or ask for a separate indentation of the text In.</p>
<p><code>empty-cells</code> It helps us control the borders of empty cells.</p>
<p><code>border-spacing</code> to control the distance between cells; corresponding <code>border-collaspe</code> Controls whether the cell borders of the table need to be combined.</p>
<h3>Form</h3>
<p>We generally recommend using the finished CSS file to control the various buttons of the form, and they are also covered by common front-end frames.</p>
<h2>Characteristics of HTML5</h2>
<p>HTML5 uses a new structure of defined page elements, making our web structure more standardized and very helpful in designing the layout of the page. It's an effective substitute for the former division.<code>&lt;div&gt; &lt;/div&gt;</code> We used to use it together. <code>id,class</code> to distinguish the elements on the page.</p>
<p>We use <code>&lt;header&gt;</code> Blocks and<code>&lt;footer&gt;</code> Blocks as home header and home footer; we can also<code>article</code> Use in block<code>&lt;header&gt;</code> Blocks and<code>&lt;footer&gt;</code> Blocks as headers and footers for articles.</p>
<p>Use <code>&lt;nav&gt;</code> Block defines the main navigator for the entire site;</p>
<p><code>&lt;article&gt;</code> The blocks make up all the components of the entire page, and they can be used either individually or embedded, which is the most extensive structure used for the construction of the entire page.</p>
<p><code>&lt;aside&gt;</code> For provision<code>&lt;article&gt;</code> A piece of subsidiary information when he's not at all.<code>&lt;article&gt;</code> This provides information on the entire page, usually with an external link.</p>
<p><code>&lt;section&gt;</code> To divide pages into parts, many can be included<code>&lt;article&gt;</code> It's fine.<code>&lt;article&gt;</code> Splits articles in blocks.</p>
<p><code>&lt;hgroup&gt;</code> For the construction of headings and sub-headings, synthesizing different levels of titles.</p>
<p><code>&lt;div&gt;</code> We can still use it to assemble the entire page, and he can still use it when the current way of doing things doesn't work.</p>
<p>We don't need separate content to organize the main content of the page except<code>&lt;header&gt;</code> Blocks and<code>&lt;footer&gt;</code> Blocks <code>&lt;aside&gt;</code> It's all about the page.</p>
<p>We used to use it. <code>&lt;a hred = &quot;xxxx&quot;&gt; text &lt;/a&gt;</code> As a basic link; now HTML5 allows <code>text</code> The content is a piece, the whole piece becomes linked.</p>
