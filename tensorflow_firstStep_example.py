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
      shuffle: True or False. Whether to shuffle the data. - 是否混淆处理
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely - 是不是需要把原来的数据再训练一次
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

    # make_one_shot_iterator()表示从dataset中实例化了一个Iterator
    # get_next()表示从iterator里取出一个元素
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
# 重新建立索引
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe['median_house_value'] /= 1000.0

def train_model(learning_rate, steps, batch_size, input_feature = "total_rooms"):
    """Trains a linear regression model of one feature.

      Args:
        learning_rate: A `float`, the learning rate.
        steps: A non-zero `int`, the total number of training steps. A training step
          consists of a forward and backward pass using a single batch.
        batch_size: A non-zero `int`, the batch size.
        input_feature: A `string` specifying a column from `california_housing_dataframe`
          to use as input feature.
      """

    peroids = 10
    steps_per_period = steps / peroids

    # feature and targets
    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]]
    my_label = "median_house_value"
    targets = california_housing_dataframe[my_label]

    feature_columns = [tf.feature_column.numeric_column(my_feature)]

    training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size=batch_size)
    prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

    # 使用梯度下降算法做的优化器
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns = feature_columns,
        optimizer = my_optimizer
    )

    plt.figure(figsize=(15, 6)) # 15cm*6cm
    plt.subplot(1, 2, 1)    # subplot子图，使用这个函数的重点是将多个图像画在同一个绘画窗口.参数分别代表：行、列、第几个
    plt.title("Learned Line by Period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    sample = california_housing_dataframe.sample(n=300) # 取样300个
    plt.scatter(sample[my_feature], sample[my_label])   # 画三点图
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, peroids)]
    # 颜色调制，numpy.linspace(start, stop, num)，从start到stop之间，走num步

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print "Training model..."
    print "RMSE (on training data):"
    root_mean_squared_errors = []
    for period in range(0, peroids):
        # 优化器开始训练，其中input_fn是要传入的函数，理论上函数每次打乱，给他系统新的数据训练。
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])

        # Compute loss.
        root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(predictions, targets))
        # Occasionally print the current loss.
        print "  period %02d : %0.2f" % (period, root_mean_squared_error)
        # Add the loss metrics from this period to our list.
        root_mean_squared_errors.append(root_mean_squared_error)
        # Finally, track the weights and biases over time.
        # Apply some math to ensure that the data and line are plotted neatly.
        y_extents = np.array([0, sample[my_label].max()])

        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents,
                                          sample[my_feature].max()),
                               sample[my_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period])    # 随机颜色在这里显示
    print "Model training finished."

    # Output a graph of loss metrics over periods.
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)
    plt.show()

    # Output a table with calibration data.
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())

    print "Final RMSE (on training data): %0.2f" % root_mean_squared_error


train_model(
    learning_rate=0.00002,
    steps=500,
    batch_size=5
)

# train_model(
#     learning_rate=0.00002,
#     steps=1000,
#     batch_size=5,
#     input_feature="population"
# )
