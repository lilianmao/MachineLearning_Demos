# -*- coding: utf-8 -*-
import glob
import io
import math
import os

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

mnist_dataframe = pd.read_csv(
  io.open("mnist_train_small.csv", "r"),
  sep=",",
  header=None)

# 使用前10000个数据作为训练和验证
mnist_dataframe = mnist_dataframe.head(10000)

mnist_dataframe = mnist_dataframe.reindex(np.random.permutation(mnist_dataframe.index))
mnist_dataframe.head()

def parse_labels_and_features(dataset):
  """Extracts labels and features.

    This is a good place to scale or transform the features if needed.

    Args:
      dataset: A Pandas `Dataframe`, containing the label on the first column and
        monochrome pixel values on the remaining columns, in row major order.
    Returns:
      A `tuple` `(labels, features)`:
        labels: A Pandas `Series`.
        features: A Pandas `DataFrame`.
    """
  labels = dataset[0]

  # 提取左右行，第1列 - 第784列的数据。（dataset的第0列是结果，后面1-784列是数据）
  features = dataset.loc[:,1:784]
  # 将所有的数据规范到[0, 1]之间，也就是说除以数据中的最大值255。
  # TODO: 待验证这条注释
  features = features / 255

  return labels, features

# 前7500个数据
training_targets, training_examples = parse_labels_and_features(mnist_dataframe[:7500])
print training_examples.describe()

# 7500-10000个数据
validation_targets, validation_examples = parse_labels_and_features(mnist_dataframe[7500:10000])
print validation_examples.describe()

"""
# 显示一个随机样本及其对应的标签。
# reshape是重新展示的函数
rand_example = np.random.choice(training_examples.index)
_, ax = plt.subplots()
ax.matshow(training_examples.loc[rand_example].values.reshape(28, 28))
ax.set_title("Label: %i" % training_targets.loc[rand_example])
ax.grid(False)
plt.show()
"""

def construct_feature_columns():
  """Construct the TensorFlow Feature Columns.

    Returns:
      A set of feature columns
    """

  # There are 784 pixels in each image，也就是这个column的维度。
  return set([tf.feature_column.numeric_column('pixels', shape=784)])

def create_training_input_fn(features, labels, batch_size, num_epochs=None, shuffle=True):
  """A custom input_fn for sending MNIST data to the estimator for training.

  Args:
    features: The training features.
    labels: The training labels.
    batch_size: Batch size to use during training.

  Returns:
    A function that returns batches of training features and labels during
    training.
  """

  def _input_fn(num_epochs=None, shuffle=True):
    # Input pipelines are reset with each call to .train().
    # 为了达到很好的数据样本，甚至当步数很小的时候。我们需要在创建Dataset前混合所有的数据，也就是permutation操作。
    # reindex不是打乱，这里是打乱数据。
    idx = np.random.permutation(features.index)       # 取随机数
    raw_features = {"pixels": features.reindex(idx)}  # features按照随机数的索引reindex，建立dict。
    raw_targets = np.array(labels[idx])               # labels按照随机数的索引简历数组

    ds = Dataset.from_tensor_slices((raw_features, raw_targets))  # 建立Dataset
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
      ds = ds.shuffle(10000)

    # Return the next batch of data
    feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
    return feature_batch, label_batch

  # 返回函数本身
  return _input_fn


# 上面的train函数做随机化处理，而下面的predict函数未做随机化处理。
def create_predict_input_fn(features, labels, batch_size):
  """A custom input_fn for sending mnist data to the estimator for predictions.

  Args:
    features: The features to base predictions on.
    labels: The labels of the prediction examples.

  Returns:
    A function that returns features and labels for predictions.
  """

  def _input_fn():
    raw_features = {"pixels": features.values}
    raw_targets = np.array(labels)

    ds = Dataset.from_tensor_slices((raw_features, raw_targets))  # warning: 2GB limit
    ds = ds.batch(batch_size)

    # Return the next batch of data
    feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
    return feature_batch, label_batch

  return _input_fn


def train_linear_classification_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
  """Trains a linear classification model for the MNIST digits dataset.

  In addition to training, this function also prints training progress information,
  a plot of the training and validation loss over time, and a confusion
  matrix.（混淆矩阵）

  Args:
    learning_rate: An `int`, the learning rate to use.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing the training features.
    training_targets: A `DataFrame` containing the training labels.
    validation_examples: A `DataFrame` containing the validation features.
    validation_targets: A `DataFrame` containing the validation labels.

  Returns:
    The trained `LinearClassifier` object.
  """

  periods = 10

  steps_per_period = steps / periods
  # Create the input functions.
  predict_training_input_fn = create_predict_input_fn(
    training_examples, training_targets, batch_size)
  predict_validation_input_fn = create_predict_input_fn(
    validation_examples, validation_targets, batch_size)
  training_input_fn = create_training_input_fn(
    training_examples, training_targets, batch_size)

  # Create a LinearClassifier object.
  my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  classifier = tf.estimator.LinearClassifier(
    feature_columns=construct_feature_columns(),
    n_classes=10,
    optimizer=my_optimizer,
    config=tf.estimator.RunConfig(keep_checkpoint_max=1)
    # The maximum number of recent checkpoint files to keep.需要保存的最大数量的checkpoint文件数量。
  )

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print "Training model..."
  print "LogLoss error (on validation data):"
  training_errors = []
  validation_errors = []
  for period in range(0, periods):
    # Train the model, starting from the prior state.
    classifier.train(
      input_fn=training_input_fn,
      steps=steps_per_period
    )

    # Take a break and compute probabilities.
    training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
    training_probabilities = np.array([item['probabilities'] for item in training_predictions])
    training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
    training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)
    # 这里应该是将id取出来，变成二进制的形式，在10长度的数组中，第6个位置放个1，标记为6。（从第0个开始计算）

    validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
    validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
    validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
    validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)

    # Compute training and validation errors.
    training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
    validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
    # Occasionally print the current loss.
    print "  period %02d : %0.2f" % (period, validation_log_loss)
    # Add the loss metrics from this period to our list.
    training_errors.append(training_log_loss)
    validation_errors.append(validation_log_loss)
  print "Model training finished."
  # 移除事件文件来节省内存空间
  _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

  # Calculate final predictions (not probabilities, as above).
  final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
  final_predictions = np.array([item['class_ids'][0] for item in final_predictions])

  accuracy = metrics.accuracy_score(validation_targets, final_predictions)
  print "Final accuracy (on validation data): %0.2f" % accuracy

  # Output a graph of loss metrics over periods.
  plt.ylabel("LogLoss")
  plt.xlabel("Periods")
  plt.title("LogLoss vs. Periods")
  plt.plot(training_errors, label="training")
  plt.plot(validation_errors, label="validation")
  plt.legend()
  plt.show()

  # 输出混淆矩阵，精度评价的一种标准格式，这里下面3行都不理解？？？
  cm = metrics.confusion_matrix(validation_targets, final_predictions)
  # Normalize the confusion matrix by row (i.e by the number of samples in each class)
  cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
  ax = sns.heatmap(cm_normalized, cmap="bone_r")
  ax.set_aspect(1)    # 横纵坐标的比例
  plt.title("Confusion matrix")
  plt.ylabel("True label")
  plt.xlabel("Predicted label")
  plt.show()

  return classifier

"""
classifier = train_linear_classification_model(
             learning_rate=0.03,
             steps=1000,
             batch_size=30,
             training_examples=training_examples,
             training_targets=training_targets,
             validation_examples=validation_examples,
             validation_targets=validation_targets)
"""

def train_nn_regression_model(
        learning_rate,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a neural network classification model for the MNIST digits dataset.

    In addition to training, this function also prints training progress information,
    a plot of the training and validation loss over time, as well as a confusion
    matrix.

    Args:
      learning_rate: An `int`, the learning rate to use.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      hidden_units: A `list` of int values, specifying the number of neurons in each layer.
      training_examples: A `DataFrame` containing the training features.
      training_targets: A `DataFrame` containing the training labels.
      validation_examples: A `DataFrame` containing the validation features.
      validation_targets: A `DataFrame` containing the validation labels.

    Returns:
      The trained `DNNClassifier` object.
    """

    periods = 10
    # Caution: input pipelines are reset with each call to train.
    # If the number of steps is small, your model may never see most of the data.
    # So with multiple `.train` calls like this you may want to control the length
    # of training with num_epochs passed to the input_fn. Or, you can do a really-big shuffle,
    # or since it's in-memory data, shuffle all the data in the `input_fn`.
    steps_per_period = steps / periods

    # Create input functions
    predict_training_input_fn = create_predict_input_fn(
      training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(
      validation_examples, validation_targets, batch_size)
    training_input_fn = create_training_input_fn(
      training_examples, training_targets, batch_size)

    # 创建一个DNNClassifier对象
    # 这里的n_classes就是指分成几类（这里labels总共就10个类，如果你的n_classes>10，也是没用的，目标没那么多，训练那么多有何用）
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    classifier = tf.estimator.DNNClassifier(
        feature_columns=[tf.feature_column.numeric_column('pixels', shape=784)],
        n_classes=10,
        hidden_units=hidden_units,
        optimizer=my_optimizer,
        config=tf.estimator.RunConfig(keep_checkpoint_max=1)
    )

    # Train the model, but do so inside a loop so that we can periodically assess loss metrics.
    print "Training model..."
    print "RMSE (on training data):"
    training_errors = []
    validation_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        # Take a break and compute probabilities.
        # 系统会根据可能性帮你计算好这个feature属于那个label，这里probabilities只是展示一下。
        # TODO: 争取找到无法调试的原因
        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)
        # 这里应该是将id取出来，变成二进制的形式，在10长度的数组中，第6个位置放个1，标记为6。（从第0个开始计算）

        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)

        # Compute training and validation errors.
        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
        # Occasionally print the current loss.
        print "  period %02d : %0.2f" % (period, validation_log_loss)
        # Add the loss metrics from this period to our list.
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
    print "Model training finished."
    # Remove event files to save disk space.
    _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

    # Calculate final predictions (not probabilities, as above).
    final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])

    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print "Final accuracy (on validation data): %0.2f" % accuracy

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    plt.show()

    # 输出混淆矩阵，精度评价的一种标准格式，这里下面3行都不理解？？？
    cm = metrics.confusion_matrix(validation_targets, final_predictions)
    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)  # 横纵坐标的比例
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

    return classifier

classifier =train_nn_regression_model(
             learning_rate=0.05,
             steps=1000,
             batch_size=30,
             hidden_units=[100, 100],
             training_examples=training_examples,
             training_targets=training_targets,
             validation_examples=validation_examples,
             validation_targets=validation_targets)

mnist_test_dataframe = pd.read_csv(
  io.open("mnist_test.csv", "r"),
  sep=",",
  header=None)

test_targets, test_examples = parse_labels_and_features(mnist_test_dataframe)

predict_test_input_fn = create_predict_input_fn(
      test_examples, test_targets, 100)
test_predictions = classifier.predict(input_fn=predict_test_input_fn)
test_predictions = np.array([item['class_ids'][0] for item in test_predictions])

accuracy = metrics.accuracy_score(test_targets, test_predictions)
print "ccuracy on test data: %0.2f" % accuracy
