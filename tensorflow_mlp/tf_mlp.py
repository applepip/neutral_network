import tensorflow as tf
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]

# 训练两个隐藏层的 DNN（一个具有 300 个神经元，另一个具有 100 个神经元）和一个具有 10 个神
# 经元的 SOFTMax 输出层进行分类,DNNClassifier 基于 Relu 激活函数创建所有神经元层（我们可以
# 通过设置超参数 activation_fn 来改变激活函数）。输出层基于 SoftMax 函数，损失函数是交叉熵。
dnn_clf = tf.compat.v2.estimator.DNNClassifier(hidden_units=[300,100], n_classes=10,
                                     feature_columns=feature_cols)

input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"X": X_train}, y=y_train, num_epochs=40, batch_size=50, shuffle=True)
dnn_clf.train(input_fn=input_fn)

test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"X": X_test}, y=y_test, shuffle=False)


eval_results = dnn_clf.evaluate(input_fn=test_input_fn)

print(eval_results)