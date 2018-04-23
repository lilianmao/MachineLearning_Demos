# -*- coding: utf-8 -*-
import collections
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
from sklearn import metrics
import sys

tf.logging.set_verbosity(tf.logging.ERROR)
train_url = 'https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/train.tfrecord'
train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)
test_url = 'https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/test.tfrecord'
test_path = tf.keras.utils.get_file(test_url.split('/')[-1], test_url)
# tf.keras 中包含一个文件下载和缓存工具，[-1]是list中倒数第一个的意思。

def _parse_function(record):
    """提取features和labels

    Args:
        record: File path to a TFRecord file
    Returns:
        A `tuple` `(labels, features)`:
          features: A dict of tensors representing the features标签组成的字典
          labels: A tensor with the corresponding labels.
    """

    features = {
        "terms": tf.VarLenFeature(dtype=tf.string),  # terms是变长string
        "labels": tf.FixedLenFeature(shape=[1], dtype=tf.float32)  # labels是0或1。
    }

    # parse_single_example应该是转为TFRecord文件。
    parsed_features = tf.parse_single_example(record, features)

    terms = parsed_features['terms'].values
    labels = parsed_features['labels']

    return {'terms': terms}, labels

"""
# 创建Dataset对象，以train_path路径下的数据创建。
ds = tf.data.TFRecordDataset(train_path)
# 使用_parse_function函数将数据映射到特征和标签
ds = ds.map(_parse_function)

print ds

# 从训练数据集中获取第一个样本以观察。
n = ds.make_one_shot_iterator().get_next()
sess = tf.Session()
sess.run(n)
"""

# 创建一个输入函数，该输入函数的功能是转换所给文件的tf.Examples，并且把他们分成features和targets。
def _input_fn(input_filenames, num_epochs=None, shuffle=True):
    # 创建Dataset对象，以input_filenames路径下的数据创建。
    ds = tf.data.TFRecordDataset(input_filenames)
    # apply a function to each element, the element structure determines the arguments of the function
    # 对每个元素都是用一遍这个函数，元素的结构决定了函数的参数。
    ds = ds.map(_parse_function)

    if shuffle:
        ds = ds.shuffle(10000)

    # Our feature data is variable-length, so we pad and batch
    # each field of the dataset structure to whatever size is necessary
    # Dataset.padded_batch() 转换允许你将不同shape的tensors进行batch。
    # Dataset.padded_batch() 转换允许你为每个component的每个dimension设置不同的padding，它可以是变长（在样本上指定None即可）或定度。你可以对padding值（缺省为0.0）进行override。
    # 参考：http://d0evi1.com/tensorflow/datasets/
    ds = ds.padded_batch(25, ds.output_shapes)

    ds = ds.repeat(num_epochs)

    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

# informative_terms = ("bad", "great", "best", "worst", "fun", "beautiful",
#                      "excellent", "poor", "boring", "awful", "terrible",
#                      "definitely", "perfect", "liked", "worse", "waste",
#                      "entertaining", "loved", "unfortunately", "amazing",
#                      "enjoyed", "favorite", "horrible", "brilliant", "highly",
#                      "simple", "annoying", "today", "hilarious", "enjoyable",
#                      "dull", "fantastic", "poorly", "fails", "disappointing",
#                      "disappointment", "not", "him", "her", "good", "time",
#                      "?", ".", "!", "movie", "film", "action", "comedy",
#                      "drama", "family", "man", "woman", "boy", "girl")

informative_terms = None;

with open("terms.txt", 'r') as f:
    informative_terms = list(set(f.read().split()))
# 前3w个数据，每个300个去一个，Python切片。
informative_terms = informative_terms[:30000:300]


# categorical_column_with_vocabulary_list()可以根据显示词汇表把每个字符串映射到一个整数上。
terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key="terms", vocabulary_list=informative_terms)

my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
# clip_gradients_by_norm()的作用是使权重的更新限制在一个合适的范围，防止过大或者过小。


"""
# 1. 线性回归
classifier = tf.estimator.LinearClassifier(
    feature_columns=[terms_feature_column],
    optimizer=my_optimizer,
)


# 2. DNN
classifier = tf.estimator.DNNClassifier(
    feature_columns=[tf.feature_column.indicator_column(terms_feature_column)],
    hidden_units=[20, 20],
    optimizer=my_optimizer,
)
"""

# 3. 在DNN中使用嵌入
classifier = tf.estimator.DNNClassifier(
    # dimension指定嵌入的维度，这里维度是2是为了让我们可以直观的感觉出来单词之间的相似度，其实我感觉dimension是2并不是最佳效果。
    feature_columns=[tf.feature_column.embedding_column(terms_feature_column, dimension=2)],
    hidden_units=[10, 10],
    optimizer=my_optimizer
)

try:
    classifier.train(
        input_fn=lambda: _input_fn([train_path]),
        steps=1000
    )

    evaluation_metrics = classifier.evaluate(
        input_fn=lambda: _input_fn([train_path]),
        steps=1000
    )
    print "Training set metrics:"
    for m in evaluation_metrics:
      print m, evaluation_metrics[m]
    print "---"

    evaluation_metrics = classifier.evaluate(
      input_fn=lambda: _input_fn([test_path]),
      steps=1000
    )
    print "Test set metrics:"
    for m in evaluation_metrics:
      print m, evaluation_metrics[m]
    print "---"
except ValueError as err:
    print err

print classifier.get_variable_names()
print classifier.get_variable_value('dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights').shape

embedding_matrix = classifier.get_variable_value('dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights')

reload(sys)
sys.setdefaultencoding('utf-8')

# 根据或得到的二维权重，将词呈现在二维空间中，通过词于词之间的关系，可以得知他们的相似度。
for term_index in range(len(informative_terms)):
    # 创建一个one_hot
    term_vector = np.zeros(len(informative_terms))
    term_vector[term_index] = 1
    # 将one-hot向量放入embedding空间
    # np.matmul是矩阵相乘，这里是一个1*54的矩阵 乘以 52*2的矩阵。
    embedding_xy = np.matmul(term_vector, embedding_matrix)
    # plt.text()根据坐标显示文字
    plt.text(
        embedding_xy[0],
        embedding_xy[1],
        informative_terms[term_index]
    )

# Do a little set-up to make sure the plot displays nicely.
plt.rcParams["figure.figsize"] = (36, 36)
plt.xlim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
plt.ylim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
plt.show()