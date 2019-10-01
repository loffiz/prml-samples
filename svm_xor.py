# -*- coding:utf-8 -*-
"""
Code for the course <<pattern recognition>> of UESTC.
Copyleft <2015,2016,2018>
hao <at> uestc.edu.cn
"""
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

X_train = X_test = X
y_train = y_test = Y

# 建立支持向量机分类器
lsvc = svm.SVC(kernel='rbf', gamma=0.01, decision_function_shape='ovr')
#lsvc = svm.SVC(kernel='linear')

lsvc.fit(X_train, y_train)

# 输出模型自带的准确性测评
print 'Accuracy：', lsvc.score(X_test, y_test)

# 对测试样本的数字类别进行预测
y_predict = lsvc.predict(X_test)
print y_predict

# 输出classification_report的预测结果分析
print classification_report(y_test, y_predict, target_names="01")

