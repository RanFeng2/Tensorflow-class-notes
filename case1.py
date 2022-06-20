import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

# 读数据和标签
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 打乱
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)  # 设置全局随机种子,因为后面还要取随机数

# 训练集+测试集
x_train = x_data[:-30]  # 除了最后30行
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

#标记为可训练y=w1*x+b1
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)


# from_tensor_slices函数使输入特征和标签值一一对应。（把数据集分批次，每个批次batch组数据）
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

epoch = 500
lr = 0.1
loss_all = 0
train_loss_results = []
test_acc = []
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):  # 需要step因此enumberate()
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)                    # 本次训练值归一
            y_ = tf.one_hot(y_train, depth=3)       # 本次实际标签值转为独热码
            loss = tf.reduce_mean(tf.square(y_-y))  # loss函数：均方误差
            loss_all += loss.numpy()
        grads = tape.gradient(loss, [w1, b1])

        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])

    print("Epoch {}, loss: {}".format(epoch, loss_all/4))
    train_loss_results.append(loss_all / 4)
    loss_all = 0

    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1  # 使用更新后的参数带入测试集预测
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)     # 返回y中最大值的索引，即预测的分类
        pred = tf.cast(pred, dtype=y_test.dtype)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)  # 判断预测是否正确
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]

    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("--------------------------")

# 绘制 loss 曲线
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Loss')  # y轴变量名称
plt.plot(train_loss_results, label="$Loss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()  # 画出曲线图标
plt.show()  # 画出图像

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()

