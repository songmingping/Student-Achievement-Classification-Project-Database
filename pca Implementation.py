import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("./CW_deleterror.csv",sep=',',header=0)
data_withoutlable=data.iloc[:,:5]


#进行手写pca降维
# (1)零均值化
def zeroMean(dataMat):
    # 求各列特征的平均值
    meanVal = np.mean(dataMat, axis=0)
    newData = dataMat - meanVal
    return newData, meanVal
newData, meanVal = zeroMean(data_withoutlable)
print ('the newData is \n', newData)
print ('the meanVal is \n', meanVal)
# （2）求协方差矩阵，rowvar=0表示每列对应一维特征
covMat = np.cov(newData, rowvar=False)
print (covMat)
# 若rowvar=1表示没行是一维特征，每列表示一个样本，显然数据不是这样的
# covMat2 = np.cov(newData, rowvar=1)
# print (covMat2)
# （3）求协方差矩阵的特征值和特征向量，利用numpy中的线性代数模块linalg中的eig函数
eigVals, eigVects = np.linalg.eig(np.mat(covMat))
print ('特征值为：\n', eigVals)
print ('特征向量为\n', eigVects)
# （4）保留主要的成分，将特征值按照从大到小的顺序排序，选择其中最大的k个，然后将对应的k个特征向量分别作为列向量组成的特征向量矩阵。
k = 2
eigValIndice = np.argsort(eigVals) # 从小到大排序
n_eigValIndice = eigValIndice[-k:] # 取值最大的k个下标
n_eigVect = eigVects[:, n_eigValIndice] # 取对应的k个特征向量
print (n_eigVect)
print (n_eigVect.shape)
lowDataMat = np.dot(newData, n_eigVect)
print(lowDataMat)


data_PCAhandWriting = pd.DataFrame(lowDataMat)

data_PCAhandWriting['label'] = data.Programme

d = data_PCAhandWriting.loc[data_PCAhandWriting.label==0.0,:]
plt.plot(d[0],d[1],'r.')
d = data_PCAhandWriting.loc[data_PCAhandWriting.label==1.0,:]
plt.plot(d[0],d[1],'y.')
d = data_PCAhandWriting.loc[data_PCAhandWriting.label==2.0,:]
plt.plot(d[0],d[1],'b.')
d = data_PCAhandWriting.loc[data_PCAhandWriting.label==3.0,:]
plt.plot(d[0],d[1],'g.')
d = data_PCAhandWriting.loc[data_PCAhandWriting.label==4.0,:]
plt.plot(d[0],d[1],'c.')
plt.show()

#发现效果比较一般

