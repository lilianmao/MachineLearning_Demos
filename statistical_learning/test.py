# -*- coding:utf-8 -*-
from numpy import *
import re
import feedparser

x = [1, 2, 3]
x1, x2, x3 = x;

print x1, x2, x3

group = array([[1,1], [2, 2], [9, 9], [10, 10]])
print group.shape[0]

martix = tile([0, 0], [4, 1])
print martix

printVec = [0]*9
print printVec, type(printVec)

mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
print mySent.split()

regEx = re.compile('\\W*')
listOfTokens = regEx.split(mySent)
print listOfTokens

print [tok.lower() for tok in listOfTokens if len(tok) > 2]

# ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
# print ny, ny['entries'], len(ny['entries'])

# append和extend用法的区别
a = [1, 2, 3]
b = [4, 5, 6]

a.append(b)
a.extend(b)

print a

def deleteList(names):
    names[1] = 'lilianmao'
    del(names[0])

names = ['lilin', 'lee']
deleteList(names=names)         # 貌似传送了地址，del对list本身操作
print names                     # 输出['lilianmao']