import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
NMF，⾮负矩阵分解，它的⽬标很明确，就是将⼤矩阵分解成两个⼩矩阵，使得这两个⼩矩阵相乘后能够还原到⼤矩阵。⽽⾮负表⽰分解的
矩阵都不包含负值。 从应⽤的⾓度来说，矩阵分解能够⽤于发现两种实体间的潜在特征，⼀个最常见的应⽤就是协同过滤中的预测打分值，
⽽从协同过滤的这个⾓度来说，⾮负也很容易理解：打分都是正的，不会出现负值。
'''

def matrix_factorisation(R, P, Q, K, steps=500, alpha=0.0002, beta=0.02):#迭代次数，学习率，控制特征向量
    # 为梯度下降常数，通常取⼀个较⼩的值（防⽌⽆法收敛），如 0.0002
    Q = Q.T
    for step in range(steps):
        #对R种元素求误差，遍历
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    #非负，对每个点求差值
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P, Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - eR[i,j], 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            break
    return P, Q.T

data = pd.read_csv("./CW_deleterror.csv",sep=',',header=0)
data1=data.iloc[:,:5]
data1=data1.values
R = data1
R = np.array(R)
N = len(R)
M = len(R[0])
K = 2
P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)
nP, nQ = matrix_factorisation(R, P, Q, K)
nR = numpy.dot(nP, nQ.T)
print(nR)
print(nP)



data_final = pd.DataFrame(nP)

data_final['label'] = data.Programme

d = data_final.loc[data_final.label==0.0,:]
plt.plot(d[0],d[1],'r.')
d = data_final.loc[data_final.label==1.0,:]
plt.plot(d[0],d[1],'y.')
d = data_final.loc[data_final.label==2.0,:]
plt.plot(d[0],d[1],'b.')
d = data_final.loc[data_final.label==3.0,:]
plt.plot(d[0],d[1],'g.')
d = data_final.loc[data_final.label==4.0,:]
plt.plot(d[0],d[1],'c.')
plt.show()