#*--coding:utf8 --*

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

# 设置固定随机种子
np.random.seed(4)

#随机生成左下方20个点，右上方20个点
X = np.r_[np.random.randn(20, 2) - [2, 2], 
		np.random.randn(20, 2) + [2, 2]]
#将前20个归为标记0，后20个归为标记1
Y = [0] * 20 + [1] * 20

#X=np.array([[0,0],[0,2],[2,0],[2,2]])
#Y=np.array([0,0,0,1])

#训练模型
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# 获取分割超平面参数
w = clf.coef_[0]  # 参数权值，由于属性只有两维，所以 w 也只有2维
b = clf.intercept_[0] 
print "超平面的系数: ", "w: ", w, "b: ", b
print "支持向量: ", clf.support_vectors_ 

# 计算与超平面平行的上下两个支撑平面
xx = np.linspace(-5, 5)  # 随机产生连续x值
yy = -(w[0] * xx + b) / w[1] #根据随机x得到y值
yy_up   = -(w[0] * xx + (b - 1)) / w[1]# 经过支持向量的点的直线
yy_down = -(w[0] * xx + (b + 1)) / w[1]# 经过支持向量的点的直线


#画出三条直线
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')  # 分界线
plt.plot(xx, yy_up, 'k--')

#画散点图
plt.scatter(clf.support_vectors_[:,0],
			clf.support_vectors_[:,1],
			s=100,
			c="g",
			facecolors='red')  #,facecolors='none',zorder=10

plt.scatter(X[:, 0],  # 样本的 x 轴数据
           X[:, 1],  # 样本集的 y 轴数据
           c=Y,  # 分类结果集
           cmap=plt.cm.Paired)  # cmap 确定颜色

plt.title('Bold circles are the support vectors')
plt.axis('tight')
plt.show()
