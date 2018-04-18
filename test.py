import numpy as np
import os
import sys

list1 = [3, 5, -4, -1, 0, -2, -6]
list2 = sorted(list1, key=lambda x: abs(x))
print list2

list3 = np.minimum(list1, 2)
print list3

judgeTrueOrFalse = lambda x: 1 if x>0 else 0;
for num in list1:
    print judgeTrueOrFalse(num)

# print (sys.path)
# print (os.env)

X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
newX = X[:, np.newaxis]
print X
print newX
print X.shape, newX.shape

train_url = 'https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/train.tfrecord'
print train_url.split('/')[-1]