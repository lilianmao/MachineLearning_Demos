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

ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
print ny, ny['entries'], len(ny['entries'])