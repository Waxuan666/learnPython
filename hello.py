import math
import copy
import functools
import time
from functools import reduce
from enum import Enum
import logging
logging.basicConfig(level=logging.INFO)
import unittest
import re
from datetime import datetime,timedelta,timezone
import base64
import hashlib
import random
import itertools
from urllib import request
import socket
import ssl
import threading
# import builtins

'''
print('hello,world')
sum = 0
for x in range(101):
    sum = sum + x
print(sum)
'''
L = ['Bart', 'Lisa', 'Adam']
# for x in L:
#    print('hello,', x)

# n = 0
# while n < 100:
#     n = n+1
#     if n % 2 == 0:
#         continue
#     print(n)
# print('END')
d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
a = d.get('Michael')
# print(a)
d['Max'] = 'Mad'
# print(d.get('Max'))
# print(max(1, 2.3))
# print(hex(4095))


def my_abs(x):
    if not isinstance(x, (int, float)):
        raise TypeError('bad operand type')
    if x >= 0:
        return x
    else:
        return -x


# print(my_abs(-18))


def move(x, y, step, angle=0):
    nx = x + step * math.cos(angle)
    ny = y + step * math.sin(angle)
    return nx, ny

# x, y = move(100,100,60,math.pi / 6)
# print(x, y)


def quadratic(a, b, c):
    x1 = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    x2 = (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
    return x1, x2


# print(quadratic(2, 3, 1))


def power(x, n=2):
    s = 1
    while n > 0:
        n = n - 1
        s = s * x
    return s


# print(power(5))


def calc(*numbers):   # 可变参数
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum


# print(calc([1, 2, 3]))
# print(calc(1, 2, 3))    # 可变参数调用，可传入任意个参数
# print(calc())

nums = [1, 2, 3]
# calc(*nums)


def person(name, age, **kw):  # 关键字参数
    if 'city' in kw:
        pass
    if 'job' in kw:
        pass
    print('name:', name, 'age:', age, 'other:', kw)


extra = {'city': 'Beijing', 'job': 'Engineer'}
# person('Jack', 24, City=extra['city'], Job=extra['job'])
# 将extra的所有key-value用关键字参数传入到函数的**kw参数，kw将获得一个拷贝extra的dict.
# person('Jack', 24, **extra)


def Person(name, age, *args, city='Shanghai', job):  # 命名关键字参数
    print(name, age, args, city, job)


# Person('Jack', 24, city='Beijing', job='Engineer')


def product(a, b, c=0, *, D, **y):
    print(a, b, c, D, y)
    # for a in y:
    #     x = x * a
    # return x


# product(1, 2, 3, 'a', 2, '3', _xx=88, _xxx='www')

args = (1, 2, 3, )  # tuple
kw = {'D': 99, 'x': '+_*/'}
# product(*args, **kw)


def fact(n):
    if n == 1:
        return 1
    return n * fact(n - 1)


# print(fact(5))


def Fact(n):
    return fact_iter(n, 1)


def fact_iter(num, product):
    if num == 1:
        return product
    return fact_iter(num - 1, num * product)


def Move(n, a, b, c):
    if n == 1:
        print(a, '-->', c)
    else:
        print(n)
        Move(n - 1, a, c, b)
        Move(1, a, b, c)
        Move(n - 1, b, a, c)


# Move(5, 'A', 'B', 'C')

'''
L1 = list(range(100))
print(L1[:20:2])
L2 = tuple(range(100))
print(L2[:10])
'''


def trim(s):
    while len(s) > 0 and s[0] == ' ':
        s = s[1:]
    while len(s) > 0 and s[-1] == ' ':
        s = s[:-1]
    return s


# print(trim('  he   llo    '))


'''
for key, value in d.items():
     print(key, value)

for ch in 'asdfghj':
    print(ch)
'''


def findMidAndMax(L):
    if L == []:
        return(None, None)
    else:
        m = M = L[0]
        for x in L:
            if x <= m:
                m = x
            if x >= M:
                M = x
        return(m, M)


# print(findMidAndMax([1]))


# print(list(range(1, 11)))

# print([x * x for x in range(1, 11)if x % 2 == 0])
# print([m + n for m in 'ABC' for n in 'XYZ'])


L1 = ['Hello', 'World', 18, 'Apple', None]   # 列表生成式
L2 = [s.lower() for s in L1 if isinstance(s, str)]


g = (x * x for x in range(10))  # generator
# for n in g:
#    print(n)


def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1
    return 'done'


# for x in fib(6):
#     print(x)


def triangles():
    La = Lb = [1, 1]
    yield [1]
    while 1:
        yield La
        x = 1
        for x in range(1, len(La)):
            Lb.insert(x, La[x - 1] + La[x])
        La = Lb[:]
        Lb = [1, 1]


'''
n = 0
results = []
for t in triangles():
    print(t)
    results.append(t)
    n = n + 1
    if n == 20:
        break
'''


def add(x, y, f):     # 编写高阶函数，就是让函数的参数能够接收别的函数。
    return f(x) + f(y)


# print(add(5, -5, abs))


def f(x):
    return x * x


r = map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9])
# print(list(map(str, [1, 2, 3, 4, 5, 6, 7, 8, 9])))


DIGITS = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
          '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}


def str2int(s):
    def fn(x, y):
        return x * 10 + y

    def char2num(s):
        return DIGITS[s]

    return reduce(fn, map(char2num, s))
# print (str2int('1234'))


def normalize(name):
    return(name[0].upper() + name[1:len(name)].lower())


L1 = ['adam', 'LISA', 'barT']
L2 = list(map(normalize, L1))
# print(L2)


def prod(L):
    return reduce(lambda x, y: x * y, L)


######## v1.0 ############################
def str2float1_0(s):
    r = reduce(lambda x, y: x * 10 + y, map(char2num, move(s)))
    x = 1
    while x < (len(s) - mark(s)):
        r = r / 10
        x += 1
    return r


DIGITS = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
          '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}


def mark(s):
    x = 1
    while x < len(s):
        if s[x] is '.':
            return x
        else:
            x = x + 1


def move(s):
    a = s.replace('.', '')
    return a


def char2num(s):
    return DIGITS[s]

#################################


# v1.1
def str2float(s):
    return reduce(lambda x, y: x * 10 + y, map(int, s.replace('.', ''))) / 10 ** (len(s) - s.find('.') - 1)


# print(str2float('123.456'))
'''
s = '123.456'
print('----------------------')
print(s)
print('s.find 查找指定字符位置 '+ str(s.find('.')))
print('s.replace 替换指定字符 '+s.replace('.', ''))
print('s.split 按字符分段 '+s.split('.')[0] + ' ' + s.split('.')[1])
'''


def odd_iter():
    n = 1
    while True:
        n = n + 2
        yield n


def indivisible(n):
    return lambda x: x % n > 0    # true:不能被当前数整除


def primes():
    yield 2
    it = odd_iter()
    while True:
        n = next(it)
        yield n
        it = filter(indivisible(n), it)  # 筛选掉能被n整除的


'''
for n in primes():
    if n < 10:
        print(n)
    else:
        break
'''

############自己写的头尾比较#################


def is_palindrome(n):
    int_str = str(n)
    x = 0
    while x < len(int_str) / 2:
        if int_str[x] == int_str[-x - 1]:
            x = x + 1
        else:
            return False
    return True


###########################################

###########[::-1]参数-1是步进，表示倒着取######


def is_palindrome_V(n):
    int_str = str(n)
    return int_str[::-1] == int_str


output = filter(is_palindrome_V, range(1, 203))
# print('1~1000:', list(output))


L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]


def sort_by_name(t):

    return t[0]


def sort_by_score(t):
    return -t[1]


L1 = sorted(L, key=sort_by_name)
L2 = sorted(L, key=sort_by_score)
# print(L1)
# print(L2)

###################闭包不要引用循环变量#################


def count():
    def f(j):
        def g():
            return j * j
        return g
    fs = []
    for i in range(1, 3):
        fs.append(f(i))  # f(i)立刻被执行，因此i的当前值被传入f()
    return fs


f1, f2 = count()
f1()


def creatCounter():
    a = [0]

    def counter():
        a[0] += 1
        return a[0]
    return counter


counterA = creatCounter()
# print(counterA(), counterA())


def is_odd(n):
    return n % 2 == 1


L = list(filter(lambda x: x % 2 == 1, range(1, 20)))
# print(L)


###################装饰器########################################
def metric(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kw):
        start = time.time()
        fn(*args, **kw)
        end = time.time()
        print('%s executed in %s ms' % (fn.__name__, end - start))
        return fn(*args, **kw)
    return wrapper
    pass


@metric
def fast(x, y):
    time.sleep(0.0012)
    return x + y
# f = fast(11, 22)
# print(f)


int2 = functools.partial(int, base=2)  # 偏函数#####


##private _XXXX or  __XXXX ############


def _private_1(name):
    return 'Hello, %s' % name


def _private_2(name):
    return 'Hi, %s' % name


def greeting(name):
    if len(name) > 3:
        return _private_1(name)
    else:
        return _private_2(name)


# print(greeting('wang'))

#######################################################


##########################    类方法  private    ##################
class Student(object):
    def __init__(self, name, gender, score):
        self.__name = name
        self.__gender = gender
        self.__score = score

    # def print_score(self):
    #    print('%s: %s' % (self.__name, self.__score))

    def get_grade(self):
        if self.__score >= 90:
            return 'A'
        elif self.__score >= 60:
            return 'B'
        else:
            return 'C'

    def get_name(self):
        return self.__name

    def get_score(self):
        return self.__score

    def set_score(self, score):
        if 0 <= score <= 100:
            self.__score = score
        else:
            raise ValueError('bad score')

    def get_gender(self):
        return self.__gender

    def set_gender(self, gender):
        if gender == 'male' or gender == 'female':
            self.__gender = gender
        else:
            raise ValueError('WTF!')


def print_score(self, name, score):
    print('%s: %s' % (name, score))


Student.print_score = print_score  # 动态语言 给class绑定方法
wang = Student('Wang', 'female', 80)
# wang.print_score()
wang.set_score(90)
wang.set_gender('male')
# wang.print_score('wang', 99)
#########################Python中，private不是绝对不能访问#####################
# ******  wang._Student__score = 80
# print(wang.get_name(), wang.get_gender(), wang.get_score(), wang.get_grade())


#############################  继承 多态  #######################################
class Animal(object):
    def __init__(self):
        pass

    def run(self):
        print('Animal is runnig!')


class cat(Animal):
    def run(self):
        print('cat is running!')

    def eat(self):
        print('Eating fish!')


class timer (object):
    def run(self):
        print('TIK TOK...')
        pass


cat1 = cat()
# cat1.run()


def run_twice(Animal):
    Animal.run()
    Animal.run()


# run_twice(cat())  # 多态
# run_twice(timer())  # 动态语言的“鸭子类型”，它并不要求严格的继承体系，只要“像鸭子”（有run方法）就可以。

######################## 类属性 与 实例属性 #####################################


class Student1(object):
    __slots__ = ('name', 'age')
    count = 0

    def __init__(self, name):
        self.name = name
        Student1.count += 1


lisa = Student1('Bart')
# print(Student1.count)
S = Student1('BoB')
S.name = 'Bob'
S.age = 20
# S.gender = 'male'   ####  AttributeError: 'Student1' object has no attribute 'gender'


class Student2(object):

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        if not isinstance(value, int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value

    @property
    def birth(self):
        return self._birth
    @birth.setter
    def birth(self, value):
        self._birth = value

    @property
    def age(self):
        return 2019 - self._birth


s1 = Student2()
s1.score = 90
# print(s1.score)

s2 = Student2()
s2.birth = 1998
# print(s2.birth)
# print(s2.age)


class Screen(object):
    @property
    def width(self):
        return self._width
    @width.setter
    def width(self, value):
        self._width = value

    @property
    def height(self):
        return self._height
    @height.setter
    def height(self, value):
        self._height = value

    @property
    def resolution(self):
        return self._width * self._height


Screen1 = Screen()
Screen1.width = 1024
Screen1.height = 768
# print(Screen1.resolution)


############### MixIn  多重继承 ####################################
class Animal(object):
    pass

# 大类:
class Mammal(Animal):
    pass


class Bird(Animal):
    pass


class RunnableMixIn(object):
    def run(self):
        print('Running...')


class FlyableMixIn(object):
    def fly(self):
        print('Flying...')

# 各种动物:
class Dog(Mammal, RunnableMixIn):
    pass


class Bat(Mammal):
    pass


class Parrot(Bird, FlyableMixIn):
    pass


class Ostrich(Bird):
    pass


Dog1 = Dog()
# Dog1.run()
# Parrot1 = Parrot()
# bParrot1.fly()


#################### 定制类  ############################


class Student3(object):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return 'Student3 object (name: %s)' % self.name


# print(Student3('Michael'))


#################### __iter__()方法,该方法返回一个迭代对象，然后，Python的for循环就会不断调用该迭代对象的
####################__next__()方法拿到循环的下一个值，直到遇到

class Fib(object):
    def __init__(self):
        self.a = 0
        self.b = 1

    def __iter__(self):
        return self

    def __next__(self):
        self.a, self.b = self.b, self.a + self.b
        if self.a > 100000:
            raise StopIteration()
        return self.a

#要表现得像list那样按照下标取出元素，需要实现__getitem__()方法：
    def __getitem__(self, n):
        if isinstance(n, int):
            a, b = 1, 1
            for x in range(n):
                a, b = b, a + b
            return a
        if isinstance(n, slice):
            start = n.start
            stop = n.stop
            if start is None:
                start = 0
            a, b = 1, 1
            L = []
            for x in range(stop):
                if x >= start:
                    L.append(a)
                    a, b = b, a + b
            return L

# for x in Fib():
#    print(x)


f = Fib()
# print(f[0])
# print(f[1:10])


######################__getattr__############################
class Student4(object):

    def __getattr__(self, attr):
        if attr=='age':
            return lambda: 25
        raise AttributeError('\'Student4\' object has no attribute \'%s\'' % attr)
# __call__()还可以定义参数。对实例进行直接调用就好比对一个函数进行调用一样，所以你完全可以把对象看成函数，
# 把函数看成对象，因为这两者之间本来就没啥根本的区别。

    def __call__(self,name):
        print('My name is %s.' % name)


s = Student4()
# print(s.age())

s = Student4()
# print(s('Kiven'))


# 通过callable()函数，我们就可以判断一个对象是否是“可调用”对象.
# 枚举类，Enum
Month = Enum('month', ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))

#for name, member in Month.__members__.items():
#   print(name, '=>', member, ',', member.value)



class Gender(Enum):
    Male = 0
    Female = 1

class Student5(object):
    def __init__(self, name, gender):
        self.name = name
        self.gender = gender


bart = Student5('Bart', Gender.Male)
#print(bart.gender)


###############  错误 调试 测试  ##################
'''
try:
    print('try...')
    r = 10 / int('5')
    print('result:', r)
except ValueError as e:
    print('ValueError:', e)
except ZeroDivisionError as e:
    print('except:', e)
else:
    print('no error!')
finally:
    print('finally...')
print('END')
'''

def foo(s):
    return 10 / int(s)

def bar(s):
    return foo(s) * 2

def main():
    try:
        bar('2')
    except Exception as e:   ###  捕获Exception里的所有错误   ######
        logging.exception(e)
        print('Error:', e)
    finally:
        print('finally...')

#main()
def foo(s):
    n = int(s)
    if n==0:
        raise ValueError('invalid value: %s' % s)
    return 10 / n

def bar():
    try:
        foo('0')
    except ValueError as e:
        print('ValueError!')
        #raise

#bar()


def foo(s):
    s = '0'
    n = int(s)
    # assert n != 0, 'n is zero!'   ###    断言
    logging.info('n = %d' % n)
    return 10 / n
def main():
    foo('0')
# main()

############################## 单元测试  ###############################
class Student6(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score
    def get_grade(self):
        if 0 <= self.score <= 100:
            if self.score >= 80:
                return 'A'
            if self.score >= 60:
                return 'B'
            return 'C'
        else:
            raise ValueError


class TestStudent(unittest.TestCase):

    def test_80_to_100(self):
        s1 = Student6('Bart', 80)
        s2 = Student6('Lisa', 100)
        self.assertEqual(s1.get_grade(), 'A')
        self.assertEqual(s2.get_grade(), 'A')

    def test_60_to_80(self):
        s1 = Student6('Bart', 60)
        s2 = Student6('Lisa', 79)
        self.assertEqual(s1.get_grade(), 'B')
        self.assertEqual(s2.get_grade(), 'B')

    def test_0_to_60(self):
        s1 = Student6('Bart', 0)
        s2 = Student6('Lisa', 59)
        self.assertEqual(s1.get_grade(), 'C')
        self.assertEqual(s2.get_grade(), 'C')

    def test_invalid(self):
        s1 = Student6('Bart', -1)
        s2 = Student6('Lisa', 101)
        with self.assertRaises(ValueError):
            s1.get_grade()
        with self.assertRaises(ValueError):
            s2.get_grade()

#if __name__ == '__main__':
#    unittest.main()

#################  文档测试  ####################
def fact(n):
    '''
    Calculate 1*2*...*n
    
    >>> fact(1)
    1
    >>> fact(10)
    3628800
    >>> fact(-1)
    Traceback (most recent call last):
    ...
    ValueError
    '''
    if n < 1:
        raise ValueError()
    if n == 1:
        return 1
    return n * fact(n - 1)


if __name__ == '__main__':
    import doctest
    doctest.testmod()


import os
#递归获取所有文件和文件夹
def getAll(path=".", res=[]):
    for item in os.listdir(path):
        if os.path.isdir(item):
            getAll(item, res)
        else:
            res.append(item)
    return res
#搜索
def search(k):
    def _match(x):
        return x.find(k) > -1
    return list(filter(_match, getAll()))

#for item in list(map(lambda x:"filename:%s abs path:%s\n" %(x,os.path.abspath(x)),search("s"))):
 #   print(item)

'''
print('Process (%s) start...' % os.getpid())

pid = os.fork()
if pid == 0:
    print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))
else:
    print('I (%s) just created a child process (%s).' % (os.getpid(), pid))
'''

######################正则表达式######################
s = r'ABC\-001'

'''
test = '010-12345'
if re.match(r'^\d{3}\-\d{3,8}$', test):
    print('ok')
else:
    print('failed')
'''

#################datetime############################
def to_timestamp(dt_str, tz_str):
    dt_str = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    tz_num = int(re.match(r'UTC(.*?)\:.*', tz_str).group(1))
    dt_str_set = dt_str.replace(tzinfo=timezone(timedelta(hours=tz_num)))
    dt_num = dt_str_set.timestamp()
    return dt_num
# print(to_timestamp('2015-6-1 08:10:30', 'UTC+7:00'))

# cday = datetime.strptime('2015-6-1 18:19:59', '%Y-%m-%d %H:%M:%S')
# print(cday)


def safe_base64_decode(s):

    if(len(s) % 4 == 1):
        return base64.b64decode(s + b'===')
    elif(len(s) % 4 == 2):
        return base64.b64decode(s + b'==')
    elif(len(s) % 4 == 3):
        return base64.b64decode(s + b'=')
    else:
        return base64.b64decode(s)


md5 = hashlib.md5()
md5.update('how to use md5 in python hashlib?'.encode('utf-8'))
#print(md5.hexdigest())

db = {
    'michael': 'e10adc3949ba59abbe56e057f20f883e',
    'bob': '878ef96e86145580c38c87f0410ad153',
    'alice': '99b1c2188db85afee403b1536010c2c9'
}


def login(user, password):
    md5 = hashlib.md5()
    md5.update(password.encode('utf-8'))
    return db[user] == md5.hexdigest()


#print(login('michael', '123456'))


'''
class User(object):
    def __init__(self, username, password):
        self.username = username
        self.salt = ''.join([chr(random.randint(48, 122)) for i in range(20)])
        self.password = get_md5(password + self.salt)


db = {
    'michael': User('michael', '123456'),
    'bob': User('bob', 'abc999'),
    'alice': User('alice', 'alice2008')
}
'''


def get_md5(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def login(username, password):
    user = db[username]
    return user.password == get_md5(password + user.salt)


def pi(N):
    return sum([4/x if x % 4 == 1 else -4/x for x in itertools.takewhile(lambda x: x < 2*N, itertools.count(1, 2))])


# print(pi(1000000))

#print(sum(x if x % 2 == 0 else -x for x in [1, 2, 3, 4, 5, 6]))

'''
#####   GET 抓取URL内容
with request.urlopen('http://news-at.zhihu.com/api/4/news/latest') as f:
    data = f.read()
    print('Status:', f.status, f.reason)
    for k, v, in f.getheaders():
        print('%s, %s' % (k, v))
    print('Data:', data.decode('utf-8'))
'''

'''
req = request.Request('http://www.douban.com/')
req.add_header('User-Agent', 'Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25')
with request.urlopen(req) as f:
    print('Status:', f.status, f.reason)
    for k, v in f.getheaders():
        print('%s: %s' % (k, v))
    print('Data:', f.read().decode('utf-8'))

############  Post  #############
from urllib import request, parse
print('Login to weibo.cn...')
email = input('Email: ')
passwd = input('Password: ')
login_data = parse.urlencode([
    ('username', email),
    ('password', passwd),
    ('entry', 'mweibo'),
    ('client_id', ''),
    ('savestate', '1'),
    ('ec', ''),
    ('pagerefer', 'https://passport.weibo.cn/signin/welcome?entry=mweibo&r=http%3A%2F%2Fm.weibo.cn%2F')
])

req = request.Request('https://passport.weibo.cn/sso/login')
req.add_header('Origin', 'https://passport.weibo.cn')
req.add_header('User-Agent', 'Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25')
req.add_header('Referer', 'https://passport.weibo.cn/signin/login?entry=mweibo&res=wel&wm=3349&r=http%3A%2F%2Fm.weibo.cn%2F')

with request.urlopen(req, data=login_data.encode('utf-8')) as f:
    print('Status:', f.status, f.reason)
    for k, v in f.getheaders():
        print('%s: %s' % (k, v))
    print('Data:', f.read().decode('utf-8'))

'''


#####利用urllib读取JSON，然后将JSON解析为Python对象：
def fetch_data(url):
    import json
    with request.urlopen(url) as f:
        return json.loads(f.read().decode('utf-8'))

'''
from xml.parsers.expat import ParserCreate

## 课堂派 操作系统原理 PBT5LC
class DefaultSaxhandler(object):
    def start_element(self, name, attrs):
        print('sax:start_element: %s, attrs: %s' % (name, str(attrs)))

    def end_element(self, text):
        print('sax:end_element: %s' % text)

xml=r<?xml version="1.0"?>
<ol>
    <li><a href="/python">Python</a></li>
    <li><a href="/ruby">Ruby</a></li>
</ol>



handler = DefaultSaxhandler()
parser = ParserCreate()
parser.StartElementHandler = handler.start_element
parser.EndElementHandler = handler.end_element
parser.CharacterDataHandler = handler.char_data
parser.Parse(xml)
'''

from html.parser import HTMLParser
from html.entities import name2codepoint


class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print('<%s>' % tag)

    def handle_endtag(self, tag):
        print('</%s>' % tag)

    def handle_startendtag(self, tag, attrs):
        print('<%s/>' % tag)

    def handle_data(self, data):
        print(data)

    def handle_comment(self, data):
        print('<!--',data, '-->')

    def handle_entityref(self, name):
        print('&%s;' % name)

    def handle_charref(self, name):
        print('&#%s;' % name)


parser = MyHTMLParser()
parser.feed('''''')

'''
from PIL import Image


im = Image.open('test.jpeg')
w, h = im.size
print('size: %sx%s' % (w, h))
im.thumbnail((w//2, h//2))
print('Resize size: %sx%s'% (w//2,h//2))
im.save('small.jpeg', 'jpeg')


from tkinter import *

class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

    def createWidgets(self):
        self.helloLabel = Label(self, text='Hello, world!')
        self.helloLabel.pack()
        self.quitButton = Button(self, text='Quit', command=self.quit)
        self.quitButton.pack()


app = Application()
# 设置窗口标题:
app.master.title('Hello World')
# 主消息循环:
app.mainloop()

'''

#s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s = ssl.wrap_socket(socket.socket())
s.connect(('www.sina.com.cn', 443))

s.send(b'GET / HTTP/1.1\r\nHost: www.sina.com.cn\r\nConnection: close\r\n\r\n')

buffer = []
while True:
    d = s.recv(1024)
    if d:
        buffer.append(d)
    else:
        break
data = b''.join(buffer)

s.close()


header, html = data.split(b'\r\n\r\n', 1)
print(header.decode('utf-8'))
# 把接收的数据写入文件:
with open('sina.html', 'wb') as f:
    f.write(html)


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('127.0.0.1', 9999))
s.listen(5)
print('Waiting for connection...')

while True:
    # 接受一个新连接:
    sock, addr = s.accept()
    # 创建新线程来处理TCP连接:
    t = threading.Thread(target=tcplink, args=(sock, addr))
    t.start()


def tcplink(sock, addr):
    print('accept new connection from %s:%S...' % addr)
    sock.send(b'Welcome!')
    while True:
        data = sock.recv(1024)
        time.sleep(1)
        if not data or data.decode('utf-8') == 'exit':
            break
        sock.send(('Hello, %s!' % data.decode('utf-8')).encode('utf-8'))
    sock.close()
    print('Connection fro %s:%s closed.' % addr)




print('clone到另一台电脑上继续工作.')




















