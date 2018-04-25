# -*- coding:utf-8 -*-

from __future__ import division     # 这里看下
import random
import numpy as np
import matplotlib.pyplot as plt

def sign(v):
    if v>=0:
        return 1
    else:
        return -1

def train(train_num, train_data, learning_rate):
    w = [0, 0]
    b = 0
    for i in range(train_num):
        x = random.choice(train_data)
        x1, x2, y = x;                          # 依次赋3个值
        if (y*(x1*w[0]+ x2*w[1] + b) <= 0):     # −yi(w⋅xi+b)判定条件
            w[0] += learning_rate*y*x1
            w[1] += learning_rate*y*x2
            b += learning_rate*y
    return w, b

# 感知机的对偶形式
def train_pro(train_num, train_data, learning_rate):
    w = 0.0
    b = 0
    data_len = len(train_data)
    alpha = [0 for i in range(data_len)]    # αi
    train_array = np.array(train_data)      # list转换为array，准备相乘
    gram = np.matmul(train_array[:, 0:-1], train_array[:, 0:-1].T) # 行全取，列从0取到倒数第二个。得到一个8*8的矩阵。
    for index in range(train_num):
        temp=0
        i=random.randint(0, data_len-1)
        yi = train_array[i, -1]             # 这里不是取范围哦，是取值。
        for j in range(data_len):
            temp += alpha[j] * train_array[j, -1] * gram[j, i]
            # 作者写的是gram[i, j]，我认为是gram[j, i]，效果一样
            # 参考：https://blog.csdn.net/winter_evening/article/details/70196040
        temp += b
        if (yi * temp <= 0):
            alpha[i] += learning_rate
            b += learning_rate * yi
    for i in range(data_len):
        w += alpha[i] * train_array[i, 0:-1] * train_array[i, -1] # 取x向量和y值，再乘以alpha，加起来得到w。
    return w, b


def show(train_data, w, b):
    plt.figure()
    x1 = np.linspace(0, 8, 100)     # 0-8之间，分成100份。list
    x2 = (-b - w[0] * x1) / w[1]    # 由w[0]*x1+w[1]*y+b = 0解得，我们求得的w是各个维度的系数。
    plt.plot(x1, x2, color='r', label='y1 data')
    datas_len = len(train_data)
    for i in range(datas_len):
        if (train_data[i][-1] == 1):
            plt.scatter(train_data[i][0], train_data[i][1], s=50)   # scatter-绘制散点图
        else:
            plt.scatter(train_data[i][0], train_data[i][1], marker='x', s=50)   # 化成x形状
    plt.show()


if __name__ == '__main__':
    positive_data = [[1, 3, 1], [2, 2, 1], [3, 8, 1], [2, 6, 1]]        # 正样本
    negative_data = [[2, 1, -1], [4, 1, -1], [6, 2, -1], [7, 3, -1]]    # 负样本
    train_data = positive_data+negative_data
    w, b = train_pro(train_num=50, train_data=train_data, learning_rate=0.01)
    show(train_data, w, b)