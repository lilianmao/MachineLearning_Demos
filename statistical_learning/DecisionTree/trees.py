# -*- coding:utf-8 -*-

from math import log
import operator
import pickle

def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    names = ['no surfacing','flippers']     #这个name是指feature的名称
    return dataSet, names

# 香浓熵，求解DataSet最后一列的熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    NameCounts = {}
    for featVec in dataSet:
        currentName = featVec[-1]
        if currentName not in NameCounts.keys():
            NameCounts[currentName] = 0
        NameCounts[currentName] += 1
    shannonEnt = 0.0
    for key in NameCounts:
        prob = float(NameCounts[key]) / numEntries
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

# 函数功能：选择信息增益值最大的feature。
# 核心：条件概率。选择一个feature，找到feature的类别个数，按照类别切分dataSet，概率*香浓熵的总和即为条件概率。
# 本系统设计最巧妙的地方就是香浓熵的函数，求熵无非就是给一个dataSet，只要Name，其他什么条件都不重要。
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1       # feature的个数
    baseEntropy = calcShannonEnt(dataSet)   # baseEntropy：H(D)
    bestInfoGain = 0.0; bestFeature = -1    # bestInfoGain：最佳信息增益值 bestFeature：最佳信息增益值对应的最佳特征
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)          # set去重，set的len即feature的类别数。
        newEntropy = 0.0                    # newEntry：H(D|A)
        for value in uniqueVals:
            subDataSet = splitDataset(dataSet, i, value)    # 按照A的value给DataSet切分。
            prob = len(subDataSet) / float(len(dataSet))    # 在A的某个value在整体数据中的比例。
            newEntropy += prob * calcShannonEnt(subDataSet) # 该value情况下的香浓熵
        infoGain = baseEntropy - newEntropy # baseEntropy - newEntry = H(D) - H(D|A) = g(D, A)
        if (infoGain > bestInfoGain):       # 保存最大的信息增益值以及feature
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 选出classList中分类最多的值，这里一定是一个label，返回这个label的值。
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 创造决策树
def createTree(dataSet, names):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):     # List的第一个值的个数就等于List的长度，即类别相同，停止划分。
        return classList[0]
    if len(dataSet[0]) == 1:                                # 如果dataSet只有最后一列了，即遍历完所有特征，仍没有分清，挑选出现次数最多的类别多为返回。
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)            # 选择最好的feature，并找到feature对应的name构成字典。
    bestFeatName = names[bestFeat]
    myTree = {bestFeatName : {}}
    del(names[bestFeat])
    featValues = [example[bestFeat] for example in dataSet] # 得到元素包含的所有的属性值。
    uniqueVals = set(featValues)
    for value in uniqueVals:                                # 类似于多叉树的建立
        subNames = names[:]
        myTree[bestFeatName][value] = createTree(splitDataset(dataSet, bestFeat, value), subNames)
    return myTree

# 测试函数，尝试新的值进行划分。
def classify(inputTree, featNames, testVec):
    firstStr = inputTree.keys()[0]          # firstStr：tree的第一个key
    secondDict = inputTree[firstStr]        # secondDict：该节点下的子节点的dict
    featIndex = featNames.index(firstStr)   # featIndex：找到第几个feature
    key = testVec[featIndex]                # key：该featIndex下对应的测试数据的值
    valueOfFeat = secondDict[key]           # valueOfFeat：从secondDict中找到对应的value
    if isinstance(valueOfFeat, dict):       # 如果valueOfFeat是字典，说明还要继续。
        className = classify(valueOfFeat, featNames, testVec)
    else:                                   # 如果valueOfFeat不是字典，说明分类已经结束。
        className = valueOfFeat
    return className

def storeTree(inputTree, filename):
    fw = open(filename, 'w')
    # pickle用法详解：https://www.cnblogs.com/lincappu/p/8296078.html
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)

if __name__ == "__main__":
    dataSet, names = createDataSet()
    myTree = createTree(dataSet, names)         # 这里names传递的是地址，会因为del而改变。所以下面再次进行获取。
    dataSet, names = createDataSet()
    result = classify(myTree, names, [1, 1])
    print result

    # 文件存储
    storeTree(myTree, 'classifierStorage.txt')
    storeTree = grabTree('classifierStorage.txt')
    print storeTree