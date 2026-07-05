---
title: "Python 面向对象与装饰器：类、继承、property 与闭包"
title_en: "Python Object-Oriented Programming and Decorators: Classes, Inheritance, property, and Closures"
date: 2025-03-04 17:10:22 +0800
categories: ["Programming", "Python"]
tags: ["Learning Notes", "Python", "Object-Oriented Programming"]
author: Hyacehila
excerpt: "整理类、实例、私有化、property、slots、静态方法、类方法、继承、多态、导入类和装饰器。"
excerpt_en: "Covers classes, instances, private attributes, property, slots, static methods, class methods, inheritance, polymorphism, imports, and decorators."
mathjax: false
hidden: true
permalink: '/blog/2025/03/04/python-oop-and-decorators-learning-notes/'
---

## 类

​从这里开始，我们将进入一个在C语言没有涉及的思想 **面向对象编程**，在前面的使用中我们的程序设计是面向过程的；面向对象编程是最有效的软件编写方法之一  。

我们先编写现实世界中事务和情景的**类** **class** 然后基于这些类来创建**对象 object** 这样一大类对象自动具备这些类的通用行为 这称为 **继承 inheritance** 我们隐藏了类的内部细节只给了一些调用的函数 这叫做 **封装 encapsulation** 然后赋予对象属于自己的个性 称为 **类的特化（specialization）和泛化（generalization）**  通过**多态（polymorphism）** 实现基于对象类型的动态分派。 最后我们程序的逼真程度会达到一个高的可怕的程度

面向对象的编程思想不仅仅是为了拟真 编程就是程序员按照计算机的工作方式控制计算机完成各种任务 当我们需要开发一个复杂的系统时，代码的复杂性会让开发和维护工作都变得举步维艰 面向对象的编程思想让我们能够以更加自然的方法来处理这些问题 而封装的存在降低了开发人员的压力

​简单的说，类是对象的蓝图和模板，而对象是类的实例。当我们把一大堆拥有共同特征的对象的静态特征（属性）和动态特征（行为）都抽取出来后，就可以定义出一个叫做“类”的东西。

类是抽象的概念，而对象是具体的东西。在面向对象编程的世界中，一切皆为对象，对象都有属性和行为，每个对象都是独一无二的，而且对象一定属于某个类（型）。

### 创建和使用类
#### 定义类

``` python
class Dog:
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
        print(f"{self.name} is now sitting")
    def roll_over(self):
        print(f"{self.name} is rolled over")
#这里定义了这个类的其他方法 这些方法很明显不需要传入其他信息 也不是默认方法 他们只用一个参数 也就是实例本身
#现在可能理解的不是很透彻 不过别急 后面的内容一讲就清楚了

```
#### 创建实例
```python
my_dog = Dog('Whllie',6)
print(my_dog.name)
print(my_dog.age)
#现在我们根据前面定义的类完成了一次实例的创建 my_dog 是我们的实例名 传入了两个参数 Python执行了第一个默认方法为我们创建了my_dog.name my_dog.age 这两个变量 我们能在后面轻松的访问他 最后这个实例被返回给了my_dog 我们后面就可以用这个实例了

my_dog.name
my_dog.sit()
#这是实例的属性（在定义里面创建）与实例的方法（在后面的几个def创建） 的使用方法 句点是最常用的符号相当的重要

my_dog0 = Dog('Whllie',6)
my_dog1 = Dog('Whllie',6)
my_dog2 = Dog('Whllie',7)
#在完成类的创建以后 我们可以自由的创建实例 所有属性都一样也可以 只要存储在不同的地方
```

#### 私有化问题
```python
#对象的属性往往被希望进行安全的保存 不允许其他人员直接修改属性 而是使用方法来修改属性 此时我们需要这样操作 在属性命名的init部分中添加双下划线
class Test:

    def __init__(self, foo):
        self.__foo = foo

    def __bar(self):
        print(self.__foo)
        print('__bar')


def main():
    test = Test('hello')
    # AttributeError: 'Test' object has no attribute '__bar'
    test.__bar()
    # AttributeError: 'Test' object has no attribute '__foo'
    print(test.__foo)


if __name__ == "__main__":
    main()
```

以上代码无法正常的运行，因为定义的方法与属性都添加了双下划线，是私有的，无法直接从外部调用，只可以通过公开的方法间接调用，比如

```python
#增加一个公开的方法，从这个方法间接调用私有方法与属性
def access_private(self):
        self.__bar()
```

但是，Python并没有从语法上严格保证私有属性或方法的私密性，它只是给私有的属性和方法换了一个名字来妨碍对它们的访问，事实上如果你知道更换名字的规则仍然可以访问到它们

在实际开发中，我们并不建议将属性设置为私有的，因为这会导致子类无法访问（后面会讲到）。所以大多数Python程序员会遵循一种命名惯例就是让**属性名以单下划线开头来表示属性是受保护的**，本类之外的代码在访问这样的属性时应该要保持慎重。这种做法并不是语法上的规则，单下划线开头的属性和方法外界仍然是可以访问的，所以更多的时候它是一种暗示或隐喻
### 使用类和实例

``` python
#对类和实例的创建 访问 方法的使用我们都已经能够理解了 这里只会叙述一个例子
class Car:
    def __init__(self,make,model=tesla,year):
        self.make = make
        self.model = model
        self.year = year
        self.odometer = 0
#这个类的属性包括了四个 其中总里程是默认为0的 此时车辆型号默认为tesls
    def get_descriptive_name(self):
        long_name = f"{self.year} {self.make} {self.model}"
        return long_name
#创建了一个用来描述实例的方法
    def read_odometer(self):
        print(f"This car has {self.odometer} miles on it")
#创建了一个用来看总里程的方法
    def update_odometer(self,mileage):
        self.odometer = mileage
#创建了修改总里程的方法
    def add_odometer(self,mile):
        if mile >= 0:
        	self.odometer += mileage
        else:
            print("Error!")
#创建了添加里程数的方法
my_car = Car('audi','a4',2022)
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
```
### property装饰器
不要把属性直接保护 但是还要能够看到或者修改属性 这就是`property`装饰器的意义
```python
class Person(object):

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
        if self._age <= 16:
            print('%s正在玩飞行棋.' % self._name)
        else:
            print('%s正在玩斗地主.' % self._name)


def main():
    person = Person('王大锤', 12)
    person.play()
    person.age = 22
    person.play()
    # person.name = '白元芳'  # AttributeError: can't set attribute 我们没有授权对name属性的修改，没有对应的修改器，所以实现不了


if __name__ == '__main__':
    main()
```
### slots魔法
Python是一门[动态语言](https://zh.wikipedia.org/wiki/%E5%8A%A8%E6%80%81%E8%AF%AD%E8%A8%80)。通常，动态语言允许我们在程序运行时给对象绑定新的属性或方法，当然也可以对已经绑定的属性和方法进行解绑定。

但是如果我们需要限定自定义类型的对象只能绑定某些属性，可以通过在类中定义`__slots__`变量来进行限定。需要注意的是`__slots__`的限定只对当前类的对象生效，对子类并不起任何作用。

```python
class Person(object):

    # 限定Person对象只能绑定_name, _age和_gender属性
    __slots__ = ('_name', '_age', '_gender')

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
        if self._age <= 16:
            print('%s正在玩飞行棋.' % self._name)
        else:
            print('%s正在玩斗地主.' % self._name)


def main():
    person = Person('王大锤', 22)
    person.play()
    person._gender = '男'

    # person._is_gay = True
    # 我们希望给对象增加新属性，但是slots禁止了这个操作，因此报错
    # AttributeError: 'Person' object has no attribute '_is_gay'
```
### 静态方法和类方法
之前，我们在类中定义的方法都是对象方法，也就是说这些方法都是发送给对象的消息.

实际上，我们写在类中的方法并不需要都是对象方法。有的方法需要在创建对象（不知道能不能创建成功）的时候就被调用，我们可以使用静态方法来解决这类问题

```python
from math import sqrt


class Triangle(object):

    def __init__(self, a, b, c):
        self._a = a
        self._b = b
        self._c = c

    @staticmethod
    def is_valid(a, b, c):
        return a + b > c and b + c > a and a + c > b
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
        print('无法构成三角形.')


if __name__ == '__main__':
    main()

```

Python还可以在类中定义类方法。他依赖与类本身而无须实例，用于处理一些不涉及实例的类工作
```python
from time import time, localtime, sleep


class Clock(object):
    """数字时钟"""

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
        """走字"""
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
        """显示时间"""
        return '%02d:%02d:%02d' % \
               (self._hour, self._minute, self._second)


def main():
    # 通过类方法创建对象并获取系统时间，比外部完成更加简洁
    clock = Clock.now()
    while True:
        print(clock.show())
        sleep(1)
        clock.run()


if __name__ == '__main__':
    main()
```

### 类之间的关系
简单的说，类和类之间的关系有三种：is-a、has-a和use-a关系。

- is-a关系也叫继承或泛化，比如学生和人的关系、手机和电子产品的关系都属于继承关系。
- has-a关系通常称之为关联，比如部门和员工的关系，汽车和引擎的关系都属于关联关系；关联关系如果是整体和部分的关联，那么我们称之为聚合关系；如果整体进一步负责了部分的生命周期（整体和部分是不可分割的，同时同在也同时消亡），那么这种就是最强的关联关系，我们称之为合成关系。
- use-a关系通常称之为依赖，比如司机有一个驾驶的行为（方法），其中（的参数）使用到了汽车，那么司机和汽车的关系就是依赖关系。

利用类之间的这些关系，我们可以在已有类的基础上来完成某些操作，也可以在已有类的基础上创建新的类，这些都是实现代码复用的重要手段。复用现有的代码不仅可以减少开发的工作量，也有利于代码的管理和维护，这是我们在日常工作中都会使用到的技术手段。
### 继承与多态
编写一个新的类不一定要从空白开始 我们前面就是从空白开始 现在我们来看看一个现成类的特殊版本是怎么继承的

``` python
class EletricCar(Car):
    def __init__(self,make,model,year):
        super().__init__(make,model,year)
#创建子类的时候 父类必须在当前文件中并且在子类创建之前 我们把Car类放在了ElectricCar类创建的括号里面 super函数是一个特殊函数 让我们能够调用父类的方法 此处我们调用了__init__方法 创建了一个子类 此时这个子类继承了父类的所有属性和方法 目前两者完全一样
    	self.battery_size = 60
    def describe_battery(self):
        print(self.battery.size)
#现在我们给电动汽车这个新类了一个新的属性和新的方法
	def fill_gas_tank(self):
        print("No gas tank")
#这里我们重写了原本父类的方法 其实只是写一个同名的方法 继承只继承我们想要的，当我们调用这个经过子类重写的方法时，不同的子类对象会表现出不同的行为，这个就是多态（poly-morphism）
```

面向对象的核心是对自然界存在的事务的定义 事实上非常多的事务都很复杂 如果在一个类里面定义太多的属性会让人感到崩溃并且难以使用 我们有时候会在类里面放一个类 比如电动汽车类里面的电池

``` python
class Battery:
    def __init__(self,battery_size=75):
        self.battery_size = battery_size
    def decribe_battery(self):
        print(f"size is self.battery_size")

class EletricCar(Car):
    def __init__(self,make,model,year):
        super().__init__(make,model,year)
        self.battery = Battery()
#此时我们继承了原本的Car类 添加了新的新的属性self_battery 这个属性是一个电池类的实例

self.battery.describe_battery()
#嵌套以后使用更复杂了 但是理论上更清晰了
#模拟自然物品的过程中 你已经不是在python语法层面思考力 而是现实世界的物品的从属逻辑 这种逻辑有时候没有确定的答案 要根据需求仔细考虑并抉择
```



### 导入类

正如导入函数一样 有时候我们需要把类存储到一起 在需要的时候再导入

``` python
from car import Car
#这就是从一个名为car的py文件（一个模块）中导入类	Car的方法 导入以后你就当你已经在新文件中创建了Car类

from car import Car,ElectricCar
#从一个模块里面导入多个类也是可以的 当然如果你导入的是一个子类 他父类的所有属性和方法都可以使用 这是子类自带的特点

import car
#导入整个模块也是可以的
car.Car('tesla','model 3',2019)
#导入整个模块的时候需要进行索引

from car import *
#导入所有类的时候不需要后面使用的时候进行索引 当然 我们不建议这种做法 可能会出现创建了名字一样的函数或者类

from car import ElectricCar as EC
#当然可以创建别名

#对类的存储分类其实是大型项目开发的重要一步 各个开发人员需要对整个程序的各个模块熟悉 开发基础模块的人需要准备文档来帮助其他人员理解这些模块 而不是浪费精力去阅读
```

## 装饰器（decorator）
### 什么是装饰器
Python 装饰器是一种强大且灵活的工具，它允许你在不修改原函数代码的情况下，对函数或类的功能进行扩展。

我们前面在类中讨论过的 `@property` 与 `@xxx.setter` 装饰器都是Python提前设定好的装饰器，可以用于保护属性与修改属性。`@staticmethod`装饰器与`@classmethod`用于创建静态方法与类方法，他们都是Python中提前编写好的可以直接使用的装饰器，功能也介绍清楚了。

装饰器本质上是一个可调用对象（通常是函数），它接收一个函数或类作为输入，并返回一个新的函数或类（**一个高阶的函数**）。新的函数或类通常会在原函数或类的基础上添加额外的功能。前面介绍的几个decorator都起到了这个效果。

装饰器使用 `@` 符号来应用到函数或类上，其基本语法如下：
```python
#这是定义装饰器的部分，我们后面会再介绍
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

```

当使用 `@decorator` 语法将装饰器应用到函数上时，Python 会自动将被装饰的函数作为参数传递给装饰器函数，并将装饰器函数返回的新函数赋值给原函数名。因此，调用被装饰的函数实际上是调用装饰器返回的新函数。

### 自定义装饰器与效果
用于普通函数的装饰器
```python
import time

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
        print(f"函数 {func.__name__} 执行时间: {end_time - start_time} 秒")
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
```

装饰器也可以用于类的方法，本质上与普通函数没什么区别，如下
```python
import time

def timer_decorator(func):
#为了能够足够正确的识别类方法的参数，增加了self
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        print(f"方法 {func.__name__} 执行时间: {end_time - start_time} 秒")
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
```

进一步的，用于静态方法与类方法
```python
import time

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"方法 {func.__name__} 执行时间: {end_time - start_time} 秒")
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
```

甚至可以用于类，当然这里语法稍微变化了一些，只需要接受一个类即可
```python
def add_attribute(cls):
    cls.new_attribute = "这是新添加的属性"
    return cls

@add_attribute
class MyClass:
    pass

obj = MyClass()
print(obj.new_attribute)
```
