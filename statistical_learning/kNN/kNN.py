# -*- coding:utf-8 -*-

from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# k近邻算法
# 计算距离 - 选择距离最小的k个点 - 排序标记label
def classify0(input, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]      # 读矩阵第一维的长度
    diffMat = tile(input, (dataSetSize, 1)) - dataSet
    # tile函数：按照reps的要求重复input数组，详细用法：https://www.cnblogs.com/zibu1234/p/4210521.html
    sqDiffMat = diffMat**2              # 幂运算
    sqDistances = sqDiffMat.sum(axis=1) # 使行相加
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}                     # classCount是一个dictionary，存放每个label的个数，最终取最大的。
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1    # get函数：返回指定键的值，如果值不存在返回default值。
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # sorted函数可以排序任何可排序的迭代器对象，key = operator.itemgetter(1)是指获取对象的第一个域的值进行排序，reverse是指降序
    return sortedClassCount[0][0]

# 文件输出矩阵
def file2matrix(filename):
    love_dictionary = {'largeDoses':3, 'smallDoses':2, 'didntLike':1}   # 将label标记为数字
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines, 3))       # numberOfLines行3列的0
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()                     # 去掉所有的空格
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]  # 第index行，所有列 = listFromLine的第0 - 第3列（不包括3）Python切片了解一下
        if(listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

# 归一化特征值（有些值对于整体距离影响过大）
# newValue = (oldValue - min) / (max-min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))     # shape(dataSet)更省力，维度是（1000，3）
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# 验证分类器的效果
def datingClassTest():
    hoRatio = 0.5           # 取样率为0.5
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 用第i(0<i<500)个数据进行选择，在500-1000的训练集里进行训练，k=3
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(numTestVecs))
    print errorCount

# 测试一组输入数据
def classifyPerson():
    returnList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)  # 新数据勿忘除以ranges
    print "You will probably like this person: %s" % returnList[classifierResult-1]

# 循环读出每行，将读出的数据存放在1*1024的Numpy数组中
def img2Vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

# 用k近邻算法识别手写数字
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # listdir：列出给定目录的文件名
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))                # trainingMat存放m个1024*1的训练数据
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2Vector('trainingDigits/%s' % fileNameStr)    # 通过img2Vector将32*32数据转为1*1024的数据
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2Vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount / float(mTest))

if __name__ == '__main__':
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)       # 子图行数、列数和位置
    ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
    plt.show()
    """

    # datingClassTest()
    # classifyPerson()

    handwritingClassTest()