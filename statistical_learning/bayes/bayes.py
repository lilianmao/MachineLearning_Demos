# -*- coding:utf-8 -*-

from numpy import *
import re
import operator
import feedparser

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
    pAbusive = sum(trainCategory) / float(numTrainDocs) # 该文档属于侮辱类的概率是0.5。
    p0Num = ones(numWords); p1Num = ones(numWords)      # 如果是0，显然不合理，最后可能出现log0的情况。
    p0Denom = 2.0; p1Denom = 2.0                        # 这里为啥是2.0看不懂？？？
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 当你在C1这个类别的时候，有很多feature，每一个feature导致了一次label的值
            # 所以计算公式是：feature组成的矩阵 / 总和，即每个feature在C1类别时的条件概率
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)                       # log化，防止数据过于密集，分散数据。
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

# 划分函数
# vec2Classify是需要分类的变量，需要分类的分量*p0Vec = 选取出要分类的分量的各自概率，各自概率相乘的积得到的是p(w|Ci)
# 这里求的是各个分量各自概率的总和。是因为之前已经进行过log，log(p(w1|Ci) * p(w2|Ci)) = log(p(w1|Ci)) + log(p(w2|Ci))
# 最终：log(p(w|Ci)*p(Ci)) = log(p(w|Ci)) + log(p(Ci))
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

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

# 词袋模型，针对setOfWords2Vec的修改。
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 处理文本，小写化，去掉小于2长度的字符串。
def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
    # 统一大小写形式，方便存储。且每个都要大于2。

# 使用朴素贝叶斯进行垃圾邮件的过滤
def spamTest():
    # docList：文本list，classList：label的list。
    docList = [];   classList = []; fullText = [];
    for i in range(1, 26):
        wordList = textParse(open('spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)   # extend()：在列表末尾一次性追加另一个序列中的多个值。
        classList.append(1)
        wordList = textParse(open('ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50)             # range(50) 从0到50的list，不包括50。
    testSet = []                        # 随机选10个放入testSet中。
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = [];  trainClasses = []   # 构建输入矩阵和输出矩阵。输入数据需要矩阵化。
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))    # train
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error",docList[docIndex]
    print 'the error rate is:', float(errorCount) / len(testSet)

# 计算出现频率最高的30个词
def calcMostFreq(vocabList, fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) #迭代器第一个降序排列
    return sortedFreq[:30]

def localWords(feed1, feed0):
    docList = [];   classList = [];     fullText = []
    minLen = min(len(feed1['entries']), feed0['entries'])
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = range(2*minLen)
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ', float(errorCount) / len(testSet)
    return vocabList, p0V, p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]


if __name__ == "__main__":
    # testingNB()       # 文档分类
    # spamTest()        # 判断垃圾邮件

    # TODO 网址没数据，该函数没有成功运行
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    localWords(ny, sf)  # 从个人广告中获取区域倾向

