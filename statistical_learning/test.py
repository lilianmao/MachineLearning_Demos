# -*- coding:utf-8 -*-
from numpy import *

x = [1, 2, 3]
x1, x2, x3 = x;

print x1, x2, x3

group = array([[1,1], [2, 2], [9, 9], [10, 10]])
print group.shape[0]

martix = tile([0, 0], [4, 1])
print martix

printVec = [0]*9
print printVec, type(printVec)