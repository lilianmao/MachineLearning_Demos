# -*- coding: utf-8 -*-
import collections
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
from sklearn import metrics

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

    # parse_single_example应该是转为TFRecord文件。不甚理解？？？
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
    # Same code as above; create a dataset and map features and labels
    ds = tf.data.TFRecordDataset(input_filenames)
    ds = ds.map(_parse_function)

    if shuffle:
        ds = ds.shuffle(10000)

    # Our feature data is variable-length, so we pad and batch
    # each field of the dataset structure to whatever size is necessary
    # 我们feature的数据是变长的，所以我们使用padded_batch。不甚理解？？？
    # padded_shapes是一个list，和parse example出来要一致，给出的padded_shapes要满足VarLen的最大长度
    # 参考：http://www.jksoftcn.com/
    ds = ds.padded_batch(25, ds.output_shapes)

    ds = ds.repeat(num_epochs)

    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

informative_terms = ("bad", "great", "best", "worst", "fun", "beautiful",
                     "excellent", "poor", "boring", "awful", "terrible",
                     "definitely", "perfect", "liked", "worse", "waste",
                     "entertaining", "loved", "unfortunately", "amazing",
                     "enjoyed", "favorite", "horrible", "brilliant", "highly",
                     "simple", "annoying", "today", "hilarious", "enjoyable",
                     "dull", "fantastic", "poorly", "fails", "disappointing",
                     "disappointment", "not", "him", "her", "good", "time",
                     "?", ".", "!", "movie", "film", "action", "comedy",
                     "drama", "family", "man", "woman", "boy", "girl")

# categorical_column_with_vocabulary_list 函数可使用“字符串-特征矢量”映射来创建特征列。
terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key="terms", vocabulary_list=informative_terms)

my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

feature_columns = [terms_feature_column]


classifier = tf.estimator.LinearClassifier(
  feature_columns=feature_columns,
  optimizer=my_optimizer,
)

classifier.train(
  input_fn=lambda: _input_fn([train_path]),
  steps=1000)

evaluation_metrics = classifier.evaluate(
  input_fn=lambda: _input_fn([train_path]),
  steps=1000)
print "Training set metrics:"
for m in evaluation_metrics:
  print m, evaluation_metrics[m]
print "---"

evaluation_metrics = classifier.evaluate(
  input_fn=lambda: _input_fn([test_path]),
  steps=1000)

print "Test set metrics:"
for m in evaluation_metrics:
  print m, evaluation_metrics[m]
print "---"