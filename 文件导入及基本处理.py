import pandas as pd
import matplotlib.pylab as plt
import numpy as np


df = pd.read_csv("./CW_Data.csv",sep=',',header=0)
print(type(df))
print(type(df.values))
print(df)
print(df.values.shape)


df.drop_duplicates(keep=False,subset=['Q1','Q2','Q3','Q4','Q5'],ignore_index=True,inplace=True)
data=df.values
print(data)
print(df.isnull())
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
print(df.values)
df.drop('ID',axis=1,inplace=True)
print(df)
print(df.describe())
data=df.values


Class_0 = []
Class_1 = []
Class_2 = []
Class_3 = []
Class_4 = []
for i in range(len(data)):
    if data[i,-1]==0:
        Class_0.append(data[i,1:-1])
    elif data[i,-1]==1:
        Class_1.append(data[i, 1:-1])
    elif data[i,-1] == 2:
        Class_2.append(data[i, 1:-1])
    elif data[i,-1] == 3:
        Class_3.append(data[i, 1:-1])
    elif data[i,-1] == 4:
        Class_4.append(data[i, 1:-1])




plt.figure()
plt.title('data number')
plt.bar(['0','1','2','3','4'],[len(Class_0),len(Class_1),len(Class_2),len(Class_3),len(Class_4)],color=['deepskyblue','skyblue','dodgerblue','paleturquoise','cyan'])
plt.show()

#得分均值
plt.figure()
plt.title('average')
plt.scatter(np.array(Class_0).mean(-1),np.zeros(len(Class_0)))
plt.scatter(np.array(Class_1).mean(-1),np.zeros(len(Class_1))+1)
plt.scatter(np.array(Class_2).mean(-1),np.zeros(len(Class_2))+2)
plt.scatter(np.array(Class_3).mean(-1),np.zeros(len(Class_3))+3)
plt.scatter(np.array(Class_4).mean(-1),np.zeros(len(Class_4))+4)
plt.show()
#
# #标准差
plt.figure()
plt.title('standard deviation')
plt.scatter(np.array(Class_0).std(-1),np.zeros(len(Class_0)))
plt.scatter(np.array(Class_1).std(-1),np.zeros(len(Class_1))+1)
plt.scatter(np.array(Class_2).std(-1),np.zeros(len(Class_2))+2)
plt.scatter(np.array(Class_3).std(-1),np.zeros(len(Class_3))+3)
plt.scatter(np.array(Class_4).std(-1),np.zeros(len(Class_4))+4)
plt.show()