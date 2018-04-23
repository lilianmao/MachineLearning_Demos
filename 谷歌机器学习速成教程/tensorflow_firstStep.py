# -*- coding: utf-8 -*-
import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model - SGD算法中的随机选择样本的数量
      shuffle: True or False. Whether to shuffle the data. - 是否混乱处理
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely - 是不是需要把原来的数据在训练一次
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # 将pandad的DataFrame转为字典，字典的value是rooms的数组
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)  # 10000的数据重新洗牌

    # Return the next batch of data
    # get_next()表示从iterator里取出一个元素
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

tf.logging.set_verbosity(tf.logging.ERROR)  # 日志级别设置成 ERROR，避免干扰
pd.options.display.max_rows = 10            # 最多显示10行
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index)
)
california_housing_dataframe['median_house_value'] /= 1000.0
print california_housing_dataframe
print "----------"

print california_housing_dataframe.describe()
print "----------"

my_feature = california_housing_dataframe[['total_rooms']]  # 定义输入特征，一个中括号输出的是Series，两个中括号输出的是DataFrame
feature_columns = [tf.feature_column.numeric_column("total_rooms")] # 一个名为total_rooms的数字类型的list

targets = california_housing_dataframe["median_house_value"] # 定义标签

# 使用 LinearRegressor 配置线性回归模型，即一个优化器。
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
# 通过 clip_gradients_by_norm 将梯度裁剪应用到我们的优化器。梯度裁剪可确保梯度大小在训练期间不会变得过大，梯度过大会导致梯度下降法失败。
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# 配置线性回归模型
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns = feature_columns,
    optimizer = my_optimizer
)

_ = linear_regressor.train(
    input_fn = lambda : my_input_fn(my_feature, targets),
    steps = 100
)

# Create an input function for predictions.
# Note: Since we're making just one prediction for each example, we don't
# need to repeat or shuffle the data here.
prediction_input_fn =lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# Call predict() on the linear_regressor to make predictions. - 用原来的数据predict一次
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# for item in predictions 或得到item，取出item中Key为predictions的value的第0位，组成一个array。
predictions = np.array([item['predictions'][0] for item in predictions])

# Print Mean Squared Error and Root Mean Squared Error. - 计算方差（很奇怪，predictions是array，targets是Series）
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print "Mean Squared Error (on training data): %0.3f" % mean_squared_error
print "Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error
print "----------"

min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print "Min. Median House Value: %0.3f" % min_house_value
print "Max. Median House Value: %0.3f" % max_house_value
print "Difference between Min. and Max.: %0.3f" % min_max_difference
print "Root Mean Squared Error: %0.3f" % root_mean_squared_error
print "----------"

calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
print calibration_data.describe()
print "----------"
# 了解describe各个参数的含义 mean:均值

sample = california_housing_dataframe.sample(n=300)

x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias
print "x_0:%0.3f y_0:%0.3f" %(x_0, y_0)
print "x_1:%0.3f y_1:%0.3f" %(x_1, y_1)
print "weight:%0.3f bias:%0.3f" %(weight, bias)
print "----------"

plt.plot([x_0, x_1], [y_0, y_1], c='r')

plt.ylabel("median_house_value")
plt.xlabel("total_rooms")

plt.scatter(sample["total_rooms"], sample["median_house_value"])

plt.show()
