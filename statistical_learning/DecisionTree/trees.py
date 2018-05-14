# -*- coding:utf-8 -*-

from math import log
import operator

def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing','flippers']
    return dataSet, labels

# 香浓熵，求解DataSet最后一列的熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

# 对dataSet进行划分。划分的依据：某个feature的某个值。某个feature对应着某一列。
def splitDataset(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]         # 从第0列开始到axis列之前
            reducedFeatVec.extend(featVec[axis+1:]) # 从第axis+1列到最后
            retDataSet.append(reducedFeatVec)
    return retDataSet

if __name__ == "__main__":
    dataSet, labels = createDataSet()
    shannonEnt = calcShannonEnt(dataSet)
    print splitDataset(dataSet, 0, 1)