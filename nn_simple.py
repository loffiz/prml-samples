# -*- coding:utf-8 -*-
"""
Code for the course <<pattern recognition>> of UESTC.
Copyleft <2015,2016,2018>
hao <at> uestc.edu.cn
"""
import numpy as np

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return x * (1 - x)

""" Model class 
"""
class NeuralNetwork:
    def __init__(self, layers=[]):
        """
        :param layers: 网络结构参数列表，仅支持3层网络（一个隐含层）
        """
        self.layers = layers
        if len(self.layers) != 3:
            raise RuntimeError('Error parameters for layers.')

        np.random.seed(1)
        self.W1 = np.random.random((layers[0] + 1, layers[1] + 1)) * 2 - 1
        self.W2 = np.random.random((layers[1] + 1, layers[2])) * 2 - 1

    def __forward(self, X):
        """
        信息前向传播
        """
        L1 = sigmoid(np.dot(X, self.W1))
        L2 = sigmoid(np.dot(L1, self.W2))

        return L1, L2
 
    def fit(self, X, Y, epochs=10000, lr=0.11):
        """ 
        Training process
        """
        if X.shape[1] != self.layers[0] or Y.shape[1] != self.layers[2]:
            raise RuntimeError('Error size for training data.')

        # 构造增广样本
        X = np.hstack((X, np.ones([X.shape[0], 1])))
        
        # 批量样本训练更新
        for k in range(epochs):

            #在训练集中随机选取一行(一个数据)：randint()在范围内随机生成一个int类型
            i = np.random.randint(X.shape[0])
            x = [X[i]]
            #转为二维数据：由一维一行转为二维一行
            x = np.atleast_2d(x)

            # 获取前向传播各层输出
            L1, L2 = self.__forward(x)

            # 误差反向传播过程
            delta2 = (Y[i]-L2) * dsigmoid(L2) # L2层误差
            delta1 = delta2.dot(self.W2.T) * dsigmoid(L1) # L1层误差

            #权值更新W的改变与V的改变
            self.W1 += lr * x.T.dot(delta1)
            self.W2 += lr * L1.T.dot(delta2)

            # 输出误差
            if k % 1000 == 0:
                L1, L2 = self.__forward(x)
                #print "[Error: ]", np.mean(np.abs(Y - L2))

    def predict(self, x):
        """
        预测新的样本
        """
        x = np.hstack((np.array(x), [1]))
        x = np.atleast_2d(x)
        _, output = self.__forward(x)

        return output[0]

if __name__ == "__main__":

    nn = NeuralNetwork([2, 4, 1])

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    nn.fit(X, Y, 10000, lr=0.30)
    print "W1: \n", nn.W1
    print "W2: \n", nn.W2

    print "Predict:"
    for x in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print x, '=>', nn.predict(x)


