# -*- coding:utf-8 -*-
"""
Code for the course <<pattern recognition>> of UESTC.
Copyleft <2015,2016,2018>
hao <at> uestc.edu.cn
"""
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from sklearn import svm

# 加载手写体数字的数码图像数据
digits = load_digits()
#print digits.data.shape
#print digits.data
#print digits.target

# 随机选取75%的数据作为训练样本；其余25%的数据作为测试样本。
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)
print "Number for training: %s" %y_train.shape
print "Number for testing: %s" %y_test.shape

# 对训练和测试的特征数据进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 建立支持向量机分类器
lsvc = svm.SVC(kernel='linear')
#lsvc = svm.SVC(kernel='rbf')
# 训练模型
import time
t0 = time.clock()
print('开始训练...')

lsvc.fit(X_train, y_train)

print '训练结束', (time.clock() - t0), 's'

# 输出模型自带的准确性测评
print 'Accuracy：', lsvc.score(X_test, y_test)

# 对测试样本的数字类别进行预测
y_predict = lsvc.predict(X_test)

# 输出混淆矩阵
labels1 = list(set(y_predict))
conf_mat1 = confusion_matrix(y_test, y_predict, labels=labels1)
print "混淆矩阵\n", conf_mat1

# 输出classification_report的预测结果分析
print classification_report(y_test, y_predict, target_names=digits.target_names.astype(str))

#############################################训练模型保存、载入、使用#######################
# 保存模型
#from sklearn.externals import joblib
#joblib.dump(lsvc , 'lsvc.pkl')
#载入保存的模型
#lsvc2 = joblib.load('lsvc.pkl')

##预测
#y_pred = lsvc2.predict(X_test)

#print y_pred
