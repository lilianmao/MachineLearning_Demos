# -*- coding:utf-8 -*-

from numpy import *

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]   # 1代表侮辱性文字，0代表正常
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    # set函数用以去重，详细使用方法：http://www.runoob.com/python/python-func-set.html
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 返回词汇表中的单词是否在输入文档中出现
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)      # len个0组成的list[0, 0, ... ,0]
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec

# 初级训练
# trainMatrix是词汇矩阵，用0或1的矩阵表示词汇。trainCategory是label。
# 我们只需要计算p(w | Ci)以及p(Ci)
# 最后这里我们可以直接比较，可能的原因是pAb是0.5，两个类别结果相同，可以直接比较他们概率，以确定是不是侮辱性词语。
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs) # 该文档属于侮辱类的概率是0.5，两边相等。
    p0Num = ones(numWords); p1Num = ones(numWords)      # 如果是0，显然不合理，最后可能出现log0的情况。
    p0Denom = 2.0; p1Denom = 2.0                        # 这里为啥是2.0看不懂？？？
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 当你在C1这个类别的时候，有很多feature，（feature组成的矩阵 / 总和）
            # 即每个feature在C1类别时的条件概率
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)                       # log化，防止数据过于密集
    p0Vect = log(p0Num / p0Denom)                       # 这里这个总和看不懂？？？
    return p0Vect, p1Vect, pAbusive

# 划分函数
# vec2Classify是需要分类的变量
# 需要分类的分量 * 各个分量的概率的总和，是已经进行过log的，两个log的相加等于相乘，log(p(w|Ci)*p(Ci)) = log(p(w|Ci)) + log(p(Ci))
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))

    # test1
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classfied as:', classifyNB(thisDoc, p0V, p1V, pAb)

    # test2
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)


if __name__ == "__main__":
    testingNB()

