# -*- coding:utf-8 -*-
"""
Code for the course <<pattern recognition>> of UESTC.
Copyleft <2015,2016,2018>
hao <at> uestc.edu.cn
"""

from sklearn import tree
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pydotplus

# 加载手写体数字的数码图像数据
digits = load_digits()

# 随机选取75%的数据作为训练样本；其余25%的数据作为测试样本。
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)
print "Number for training: %s" %y_train.shape
print "Number for testing: %s" %y_test.shape

# 对训练和测试的特征数据进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

clf = tree.DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train, y_train)

score = clf.score(X_test,y_test)
print('\n[%.4f] : Accuracy of %s.' %(score, str(clf))) 

y_predict = clf.predict(X_test)
print '\n[classification_report]'
print classification_report(y_test, y_predict, target_names=digits.target_names.astype(str))

# dot_data = tree.export_graphviz(clf, out_file=None) 
# graph = pydotplus.graph_from_dot_data(dot_data)  

# graph.write_png("tree.png")	# 生成png文件 