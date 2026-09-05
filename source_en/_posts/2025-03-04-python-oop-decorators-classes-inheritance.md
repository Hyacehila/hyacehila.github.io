---
title: 'Python Object-Oriented Programming and Decorators: Classes, Inheritance, property, and Closures'
title_zh: Python 面向对象与装饰器：类、继承、property 与闭包
date: 2025-03-04 17:10:22 +0800
categories:
- Programming
- Programming Languages
tags:
- Python
- Object-Oriented Programming
author: Hyacehila
mathjax: false
hidden: true
excerpt: Covers classes, instances, private attributes, property, slots, static methods, class methods, inheritance, polymorphism,
  imports, and decorators.
description: Covers classes, instances, private attributes, property, slots, static methods, class methods, inheritance, polymorphism,
  imports, and decorators.
excerpt_zh: 整理类、实例、私有化、property、slots、静态方法、类方法、继承、多态、导入类和装饰器。
permalink: /blog/2025/03/04/python-oop-and-decorators-learning-notes/
lang: en
translation_key: 2025-03-04-python-oop-decorators-classes-inheritance
translation_status: machine
translation_source_hash: 87276cfab108ecd416921eacbe83c96e30d768cce5cdbb42a0ad539a4aec9f5e
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Category</h2>
<p>The questions in this article can also be addressed<a href="/en/blog/2023/03/18/python-basics-learning-notes/">Python Foundation: Syntax: Data Structure and File Processing</a>、<a href="/en/blog/2025/04/19/python-iterators-generators-lambda-learning-notes/">Python, Generator and Lambda: Inert Calculator and Function Tool</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>From here on, we'll enter an idea that is not covered in the C language. <strong>Object-oriented Programming</strong>In the previous use, our program design is process-oriented; object-oriented programming is one of the most effective software development methods.</p>
<p>We'll start with the real world.<strong>Category</strong> <strong>class</strong> And then we'll build them on these kinds.<strong>Object object object</strong> A large category of objects has these types of generic behaviors, which are called <strong>Succession</strong> We've hidden the internal details of the class, and we've given them only a few callable functions. <strong>Encapsulation</strong> And then you give the object your own personality, called <strong>Specialization and generalization of the category</strong>  Pass.<strong>Polymorphism</strong> (c) Achieving dynamic assignments based on the type of object. And finally, we'll have a pretty high degree of impulsiveness.</p>
<p>The object-oriented programming idea is not just for realism, but the programming idea is that the programmer controls the computer in a way that the computer does, and when we need to develop a complex system, the complexity of the code makes development and maintenance difficult, and the object-oriented programming idea allows us to deal with these problems in a more natural way, and the presence of the envelope reduces the pressure on the developers.</p>
<p>In short, the category is a blueprint and template for the object, and the target is an example of the category. When we remove a whole bunch of static (relations) and dynamic (acts) features of objects that share common features, we can define something called a class.</p>
<p>The category is abstract, and it is directed at specific things. In a world where objects are programmed, everything is objects, objects have attributes and behaviours, each object is unique and must belong to a certain category (type).</p>
<h3>Create and use class</h3>
<h4>Define Classes</h4>
<pre><code class="language-python">class Dog:
#首字母大写是一个类 这是我们的约定 这是创建类的第一行代码
    def __init__(self,name,age):
        self.name = name
        self.age = age
#类中的函数称为方法 这是我们对方法这一在全文最开始提出的概念的最后解释 函数和类是高度相似的 唯一的区别就是调用的方式（方法用点号调用）
#这里的方法是方法里最为特殊的一个（Python默认方法） 每当我们根据这个类创建新的实例时 Python会自动运行这个方法进行初始化的操作
#开头和结尾的两个下划线是对它这一特殊性的标识

#我们有三个形参 self是不能缺少的 并且一定要在最前面 实际上当Dog类被调用创建实例的时候 会自动传入self 他是一个指向实例本身的引用 让实例能够访问类中的属性和方法 所以我们不用传递self参数 传递name age就可以

#变量的定义使用了前缀self. 实际上我们是要借助实例来访问这个变量

	def sit(self):
        print(f&quot;{self.name} is now sitting&quot;)
    def roll_over(self):
        print(f&quot;{self.name} is rolled over&quot;)
#这里定义了这个类的其他方法 这些方法很明显不需要传入其他信息 也不是默认方法 他们只用一个参数 也就是实例本身
#现在可能理解的不是很透彻 不过别急 后面的内容一讲就清楚了
</code></pre>
<h4>Create instance</h4>
<pre><code class="language-python">my_dog = Dog(&#39;Whllie&#39;,6)
print(my_dog.name)
print(my_dog.age)
#现在我们根据前面定义的类完成了一次实例的创建 my_dog 是我们的实例名 传入了两个参数 Python执行了第一个默认方法为我们创建了my_dog.name my_dog.age 这两个变量 我们能在后面轻松的访问他 最后这个实例被返回给了my_dog 我们后面就可以用这个实例了

my_dog.name
my_dog.sit()
#这是实例的属性（在定义里面创建）与实例的方法（在后面的几个def创建） 的使用方法 句点是最常用的符号相当的重要

my_dog0 = Dog(&#39;Whllie&#39;,6)
my_dog1 = Dog(&#39;Whllie&#39;,6)
my_dog2 = Dog(&#39;Whllie&#39;,7)
#在完成类的创建以后 我们可以自由的创建实例 所有属性都一样也可以 只要存储在不同的地方
</code></pre>
<h4>Privatization</h4>
<pre><code class="language-python">#对象的属性往往被希望进行安全的保存 不允许其他人员直接修改属性 而是使用方法来修改属性 此时我们需要这样操作 在属性命名的init部分中添加双下划线
class Test:

    def __init__(self, foo):
        self.__foo = foo

    def __bar(self):
        print(self.__foo)
        print(&#39;__bar&#39;)


def main():
    test = Test(&#39;hello&#39;)
    # AttributeError: &#39;Test&#39; object has no attribute &#39;__bar&#39;
    test.__bar()
    # AttributeError: &#39;Test&#39; object has no attribute &#39;__foo&#39;
    print(test.__foo)


if __name__ == &quot;__main__&quot;:
    main()
</code></pre>
<p>The above code cannot run normally because both the method and the properties of the definition are underlined in a double, private, cannot be called directly from outside, but only indirectly through open means, for example</p>
<pre><code class="language-python">#增加一个公开的方法，从这个方法间接调用私有方法与属性
def access_private(self):
        self.__bar()
</code></pre>
<p>But Python does not strictly guarantee the private attributes or methods of privacy in grammatical terms, but simply changes the private attributes and methods to a name to prevent access to them, in fact, if you know the rules for changing names still allow access to them.</p>
<p>In the actual development, we do not recommend that the properties be set to private property, as this would render the subcategory inaccessible (discussed later). So most Python programmers follow a naming practice that is Jean-Claude.<strong>Attribute name starts with a single underlined to indicate that the attribute is protected</strong>, codes other than this category should be carefully viewed when accessing such attributes. This is not a grammatical rule, and the attributes and methods that begin with a single line are still accessible outside, so more often it is a hint or metaphor.</p>
<h3>Use classes and examples</h3>
<pre><code class="language-python">#对类和实例的创建 访问 方法的使用我们都已经能够理解了 这里只会叙述一个例子
class Car:
    def __init__(self,make,model=tesla,year):
        self.make = make
        self.model = model
        self.year = year
        self.odometer = 0
#这个类的属性包括了四个 其中总里程是默认为0的 此时车辆型号默认为tesls
    def get_descriptive_name(self):
        long_name = f&quot;{self.year} {self.make} {self.model}&quot;
        return long_name
#创建了一个用来描述实例的方法
    def read_odometer(self):
        print(f&quot;This car has {self.odometer} miles on it&quot;)
#创建了一个用来看总里程的方法
    def update_odometer(self,mileage):
        self.odometer = mileage
#创建了修改总里程的方法
    def add_odometer(self,mile):
        if mile &gt;= 0:
	self.odometer += mileage
        else:
            print(&quot;Error!&quot;)
#创建了添加里程数的方法
my_car = Car(&#39;audi&#39;,&#39;a4&#39;,2022)
print(my_car.get_descriptive_name())
my_car.read_odometer()
my_car.odometer = 100
my_car.read_odometer()
my_car.update_odometer(200)
my_car.read_odometer()
my_car.add_odometer(300)
my_car.read_odometer()
my_car.add_odometer(-100)
#我们更建议使用方法对属性进行修改 这样可以在构建方法的时候考虑非法输入的问题
</code></pre>
<h3>Property Decorator</h3>
<p>Do not protect the properties directly, but be able to see or modify them.<code>property</code>Meaning of the decorator</p>
<pre><code class="language-python">class Person(object):

    def __init__(self, name, age):
        self._name = name
        self._age = age

    # 访问器 - getter方法
    @property
    def name(self):
        return self._name

    # 访问器 - getter方法
    @property
    def age(self):
        return self._age
#以上的两个方法使用了property修饰，通过@property装饰的方法可以像访问属性一样调用，而不需要使用括号。这样避免了直接访问属性，实现属性的间接访问

    # 修改器 - setter方法
    @age.setter
    def age(self, age):
        self._age = age
# @age.setter：这是`@property`装饰器的配套装饰器，用于为通过@property装饰的只读属性添加设置值的功能。现在就可以像直接给属性赋值一样利用方法来修改属性了，而不是使用括号调用方法

    def play(self):
        if self._age &lt;= 16:
            print(&#39;%s正在玩飞行棋.&#39; % self._name)
        else:
            print(&#39;%s正在玩斗地主.&#39; % self._name)


def main():
    person = Person(&#39;王大锤&#39;, 12)
    person.play()
    person.age = 22
    person.play()
    # person.name = &#39;白元芳&#39;  # AttributeError: can&#39;t set attribute 我们没有授权对name属性的修改，没有对应的修改器，所以实现不了


if __name__ == &#39;__main__&#39;:
    main()
</code></pre>
<h3>Slots magic.</h3>
<p>Python is a door.<a href="https://zh.wikipedia.org/wiki/%E5%8A%A8%E6%80%81%E8%AF%AD%E8%A8%80">Dynamic language</a>I'm sorry. Usually, dynamic language allows us to bind new properties or methods to objects while running a program, and of course to untie those that have been bound.</p>
<p>But if we need to limit the self-defined type of object to bind only certain attributes, it can be defined in a category.<code>__slots__</code>variable to qualify. And what needs attention is...<code>__slots__</code>The qualification is valid only for the object of the current class and does not have any effect on the subcategory.</p>
<pre><code class="language-python">class Person(object):

    # 限定Person对象只能绑定_name, _age和_gender属性
    __slots__ = (&#39;_name&#39;, &#39;_age&#39;, &#39;_gender&#39;)

    def __init__(self, name, age):
        self._name = name
        self._age = age

    @property
    def name(self):
        return self._name

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, age):
        self._age = age

    def play(self):
        if self._age &lt;= 16:
            print(&#39;%s正在玩飞行棋.&#39; % self._name)
        else:
            print(&#39;%s正在玩斗地主.&#39; % self._name)


def main():
    person = Person(&#39;王大锤&#39;, 22)
    person.play()
    person._gender = &#39;男&#39;

    # person._is_gay = True
    # 我们希望给对象增加新属性，但是slots禁止了这个操作，因此报错
    # AttributeError: &#39;Person&#39; object has no attribute &#39;_is_gay&#39;
</code></pre>
<h3>Static and class methods</h3>
<p>Before, we defined the methods in the category as objects, which means that they were messages to objects.</p>
<p>In fact, the methods we write in the category do not need to be all the object methods. Some of the methods need to be called when creating objects (not knowing if they can be successful) and we can use static methods to solve these problems.</p>
<pre><code class="language-python">from math import sqrt


class Triangle(object):

    def __init__(self, a, b, c):
        self._a = a
        self._b = b
        self._c = c

    @staticmethod
    def is_valid(a, b, c):
        return a + b &gt; c and b + c &gt; a and a + c &gt; b
#装饰器修饰了这个方法，体现了其静态方法的属性
#a,b,c是静态方法的参数，他不需要self了，而是直接根据外部的输入判断

    def perimeter(self):
        return self._a + self._b + self._c

    def area(self):
        half = self.perimeter() / 2
        return sqrt(half * (half - self._a) *
                    (half - self._b) * (half - self._c))


def main():
    a, b, c = 3, 4, 5
    # 静态方法和类方法都是通过给类发消息来调用的
    if Triangle.is_valid(a, b, c):
        t = Triangle(a, b, c)
        print(t.perimeter())
        print(t.area())
    else:
        print(&#39;无法构成三角形.&#39;)


if __name__ == &#39;__main__&#39;:
    main()
</code></pre>
<p>Python could also define the class approach in the category. He relies on the category itself, without the need for examples, to deal with types of work that do not involve examples.</p>
<pre><code class="language-python">from time import time, localtime, sleep


class Clock(object):
    &quot;&quot;&quot;数字时钟&quot;&quot;&quot;

    def __init__(self, hour=0, minute=0, second=0):
        self._hour = hour
        self._minute = minute
        self._second = second

    @classmethod
    def now(cls):
        ctime = localtime(time())
        return cls(ctime.tm_hour, ctime.tm_min, ctime.tm_sec)
#装饰器体现了其类方法的属性，借助类方法也可以创建对象，获取信息；
#cls是类方法的第一个参数，cls代表类本身，借助 `cls` 可以访问类的属性和调用类的其他方法。后面我们就借助cls调用类实现了创建实例了工作。
    def run(self):
        &quot;&quot;&quot;走字&quot;&quot;&quot;
        self._second += 1
        if self._second == 60:
            self._second = 0
            self._minute += 1
            if self._minute == 60:
                self._minute = 0
                self._hour += 1
                if self._hour == 24:
                    self._hour = 0

    def show(self):
        &quot;&quot;&quot;显示时间&quot;&quot;&quot;
        return &#39;%02d:%02d:%02d&#39; % \
               (self._hour, self._minute, self._second)


def main():
    # 通过类方法创建对象并获取系统时间，比外部完成更加简洁
    clock = Clock.now()
    while True:
        print(clock.show())
        sleep(1)
        clock.run()


if __name__ == &#39;__main__&#39;:
    main()
</code></pre>
<h3>Relationship between categories</h3>
<p>In short, there are three types of relationships between categories: is-a, has-a and is-a.</p>
<ul>
<li>It is also called inheritance or generalization, for example, student-to-person relationships, mobile phone and electronic products.</li>
<li>Has-a relationships are often called linkages, such as those between departments and employees, cars and engines; linkages are called aggregates if they are integral and partial; and if the whole is further responsible for part of the life cycle (whole and part are indivisible, and at the same time disappears), then they are the strongest, and we call them synthetics.</li>
<li>The use-a relationship is often called dependency, for example, where the driver has a driving behaviour (method) in which (parameters) the vehicle is used, and the relationship between the driver and the car is dependency.</li>
</ul>
<p>Using these relationships between classes, we can do some of these operations on the basis of existing classes or create new ones on the basis of existing ones, which are important means of achieving re-use. Reuse of the existing code not only reduces the development workload but also facilitates code management and maintenance, which is a technical tool that we use in our daily work.</p>
<h3>Succession and polymorphism</h3>
<p>We don't have to start with blanks, we start with blanks, and now we're looking at how a special version of an off-the-shelf class is inherited.</p>
<pre><code class="language-python">class EletricCar(Car):
    def __init__(self,make,model,year):
        super().__init__(make,model,year)
#创建子类的时候 父类必须在当前文件中并且在子类创建之前 我们把Car类放在了ElectricCar类创建的括号里面 super函数是一个特殊函数 让我们能够调用父类的方法 此处我们调用了__init__方法 创建了一个子类 此时这个子类继承了父类的所有属性和方法 目前两者完全一样
	self.battery_size = 60
    def describe_battery(self):
        print(self.battery.size)
#现在我们给电动汽车这个新类了一个新的属性和新的方法
	def fill_gas_tank(self):
        print(&quot;No gas tank&quot;)
#这里我们重写了原本父类的方法 其实只是写一个同名的方法 继承只继承我们想要的，当我们调用这个经过子类重写的方法时，不同的子类对象会表现出不同的行为，这个就是多态（poly-morphism）
</code></pre>
<p>The focus of object orientation is on the definition of nature's existence, and in fact, a very large number of things are complex, and if you define too many attributes in one class that are both very crashing and difficult to use, we sometimes put a class in one of these, like batteries in electric cars.</p>
<pre><code class="language-python">class Battery:
    def __init__(self,battery_size=75):
        self.battery_size = battery_size
    def decribe_battery(self):
        print(f&quot;size is self.battery_size&quot;)

class EletricCar(Car):
    def __init__(self,make,model,year):
        super().__init__(make,model,year)
        self.battery = Battery()
#此时我们继承了原本的Car类 添加了新的新的属性self_battery 这个属性是一个电池类的实例

self.battery.describe_battery()
#嵌套以后使用更复杂了 但是理论上更清晰了
#模拟自然物品的过程中 你已经不是在python语法层面思考力 而是现实世界的物品的从属逻辑 这种逻辑有时候没有确定的答案 要根据需求仔细考虑并抉择
</code></pre>
<h3>Import Class</h3>
<p>Like import functions, sometimes we need to store the class together and import it when it's needed.</p>
<pre><code class="language-python">from car import Car
#这就是从一个名为car的py文件（一个模块）中导入类	Car的方法 导入以后你就当你已经在新文件中创建了Car类

from car import Car,ElectricCar
#从一个模块里面导入多个类也是可以的 当然如果你导入的是一个子类 他父类的所有属性和方法都可以使用 这是子类自带的特点

import car
#导入整个模块也是可以的
car.Car(&#39;tesla&#39;,&#39;model 3&#39;,2019)
#导入整个模块的时候需要进行索引

from car import *
#导入所有类的时候不需要后面使用的时候进行索引 当然 我们不建议这种做法 可能会出现创建了名字一样的函数或者类

from car import ElectricCar as EC
#当然可以创建别名

#对类的存储分类其实是大型项目开发的重要一步 各个开发人员需要对整个程序的各个模块熟悉 开发基础模块的人需要准备文档来帮助其他人员理解这些模块 而不是浪费精力去阅读
</code></pre>
<h2>Decorator</h2>
<h3>What's an decorator?</h3>
<p>Python Decorator is a powerful and flexible tool that allows you to expand the functions of functions or classes without modifying the original function code.</p>
<p>We discussed it in the class. <code>@property</code> and <code>@xxx.setter</code> Decorators are pre-set by Python to protect and modify properties.<code>@staticmethod</code>Decorator and<code>@classmethod</code>They are direct-use decorative devices developed in Python in advance and are clearly described.</p>
<p>Decorator is essentially a callable object (usually a function) which receives a function or class as input and returns a new function or class (in the case of a function or class)<strong>A high-level function</strong>I'm not sure. New functions or classes usually add additional functions to the original function or class. Several of the decoratores that were mentioned earlier have done this.</p>
<p>Decorator use <code>@</code> Symbols are applied to functions or classes, the basic syntax of which is as follows:</p>
<pre><code class="language-python">#这是定义装饰器的部分，我们后面会再介绍
def decorator(func):
    def wrapper(*args, **kwargs):
        # 在原函数执行前添加的代码
        result = func(*args, **kwargs)
        # 在原函数执行后添加的代码
        return result
    return wrapper

#常用的装饰方法
@decorator
def my_function():
    pass

#等价的装饰方法
my_function = decorator(my_function)
</code></pre>
<p>When used <code>@decorator</code> When the semantic is applied to a function, Python automatically transmits the decorative function as a parameter to the decorating function and assigns the new function returned by the decorating function to the original function name. Therefore, the decorated function is actually the new function to which the decoder returns.</p>
<h3>Custom Decorators and Effects</h3>
<p>Decorators for normal functions</p>
<pre><code class="language-python">import time

#创建装饰器函数，其本质上就是一个高阶函数，接受一个函数作为输入
def timer_decorator(func):
#在 timer_decorator 函数内部定义了一个名为 wrapper 的内部函数。
#*args是可变参数，接收任意数量的位置参数，打包成一个元组；
#**kwargs 用于接收任意数量的关键字参数，会将这些参数打包成一个字典。
    def wrapper(*args, **kwargs):
        start_time = time.time()
        #调用被装饰的函数 func，并将 *args 和 **kwargs 作为参数传递给它。将函数的返回值赋值给 result 变量，以便后续返回。
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f&quot;函数 {func.__name__} 执行时间: {end_time - start_time} 秒&quot;)
        #返回result
        return result
    #返回被包装后的函数
    return wrapper

#修饰函数，这是简洁版本语法
@timer_decorator
def add(a, b):
    return a + b

result = add(3, 5)
print(result)
</code></pre>
<p>Decorators can also be used in class methods, which are essentially no different from normal functions, as follows:</p>
<pre><code class="language-python">import time

def timer_decorator(func):
#为了能够足够正确的识别类方法的参数，增加了self
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        print(f&quot;方法 {func.__name__} 执行时间: {end_time - start_time} 秒&quot;)
        return result
    return wrapper

class MyClass:
#用前面定义的装饰器修饰了类方法
    @timer_decorator
    def my_method(self, a, b):
        return a + b

obj = MyClass()
result = obj.my_method(3, 5)
print(result)
</code></pre>
<p>Further, for static and class methods</p>
<pre><code class="language-python">import time

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f&quot;方法 {func.__name__} 执行时间: {end_time - start_time} 秒&quot;)
        return result
    return wrapper

class MyClass:
    @staticmethod
    @timer_decorator
    def static_method(a, b):
        return a + b

    @classmethod
    @timer_decorator
    def class_method(cls, a, b):
        return a + b

result1 = MyClass.static_method(3, 5)
print(result1)

result2 = MyClass.class_method(3, 5)
print(result2)
</code></pre>
<p>It could even be used in class, but the grammar has changed slightly.</p>
<pre><code class="language-python">def add_attribute(cls):
    cls.new_attribute = &quot;这是新添加的属性&quot;
    return cls

@add_attribute
class MyClass:
    pass

obj = MyClass()
print(obj.new_attribute)
</code></pre>
