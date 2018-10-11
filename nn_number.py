# -*- coding:utf-8 -*-
"""
Code for the course <<pattern recognition>> of UESTC.
Copyleft <2015,2016,2018>
hao <at> uestc.edu.cn
"""
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer #标签二值化
from sklearn.model_selection import train_test_split   #切割数据,交叉验证法
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

import nn_simple

#载入数据:8*8的数据集
digits = load_digits()
X = digits.data
Y = digits.target
#输入数据归一化：当数据集数值过大，乘以较小的权重后还是很大的数，代入sigmoid激活函数就趋近于1，不利于学习
X -= X.min()
X /= X.max()

#sklearn切分数据
X_train,X_test,y_train,y_test = train_test_split(X,Y)
print "Number for training: %s" %y_train.shape
print "Number for testing: %s" %y_test.shape

# 对训练和测试的特征数据进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#标签二值化：将原始标签(十进制)转为新标签(二进制)
labels_train = LabelBinarizer().fit_transform(y_train)

nn = nn_simple.NeuralNetwork([64,128,10])

import time
t0 = time.clock()
print('开始训练...')

i = 10
while i > 0: 
	# 可持续训练
	nn.fit(X_train, labels_train, epochs=2000, lr=0.85)

	# 评估精度
	predictions = []
	for j in range(X_test.shape[0]):
		#获取预测结果：返回与十个标签值逼近的距离，数值最大的选为本次的预测值
		o = nn.predict(X_test[j])

		#将最大的数值所对应的标签返回
		predictions.append(np.argmax(o))
	# 统计 0 和 1 的平均值结果
	accuracy = np.mean(np.equal(predictions, y_test))
	print 'Accuracy: ',accuracy
	i -= 1

print '训练结束', (time.clock() - t0), 's'

print predictions

###############模型评估#####################
#打印预测报告
print '混淆矩阵: \n', confusion_matrix(y_test, predictions)
#打印预测结果
print '结果评估: \n', classification_report(y_test, predictions)

