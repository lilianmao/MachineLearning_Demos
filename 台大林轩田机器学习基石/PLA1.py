# -*- coding: utf-8 -*-
import copy
from matplotlib import  pyplot as pl

w = [0, 0]
b = 0      #bias
yita = 0.5 #learning rate
data = [[(1, 4), 1], [(0.5, 2), 1], [(2, 2.3), 1], [(1, 0.5), -1], [(2, 1), -1], [(4, 1), -1], [(3.5, 4), 1],
        [(3, 2.2), -1]]
record = []

""" 
if y(wx+b)<=0,return false; else, return true 
"""
def sign(vec):
    global  w, b
    res = 0
    res = vec[1] * (w[0]*vec[0][0] + w[1]*vec[0][1] + b)
    if res > 0:
        return 1
    else:
        return -1;

"""
w = w + xy
"""
def update(vec):
    global  w, b, record
    w[0] = w[0] + yita*vec[1]*vec[0][0]
    w[1] = w[1] + yita*vec[1]*vec[0][1]
    b = b + yita *vec[1]
    record.append([copy.copy(w), b])

def perceptron():
    count = 1
    for d in data:
        flag = sign(d)
        if not flag > 0:
            count = 1
            update(d)
        else:
            count += 1
    if count >= len(data):
        return 1

if __name__ == "__main__":
    while 1:
        if perceptron() > 0:
            break
    print  record