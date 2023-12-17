'''
定义：高维数据转化为低维数据的过程，在此过程中可能会舍弃原有数据、创造新的变量
作用：是数据维数压缩，尽可能降低原数据的维数（复杂度），损失少量信息。
'''

import pandas as pd
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D



data = pd.read_csv("./CW_deleterror.csv",sep=',',header=0)
data1 = data.iloc[:,:5]
print(data1)
transfer= PCA(n_components=0.8)
data_new=transfer.fit_transform(data1)
print(data_new)
data_new = pd.DataFrame(data_new)
data_new['label'] = data.Programme

x = data_new.iloc[:, 0]
y = data_new.iloc[:, 1]
z = data_new.iloc[:, 2]

fig = plt.figure()
ax = Axes3D(fig)

d = data_new.loc[data_new.label==0.0,:]
ax.scatter(d[0],d[1],d[2],'r.')
d = data_new.loc[data_new.label==1.0,:]
ax.scatter(d[0],d[1],d[2],'y.')
d = data_new.loc[data_new.label==2.0,:]
ax.scatter(d[0],d[1],d[2],'b.')
d = data_new.loc[data_new.label==3.0,:]
ax.scatter(d[0],d[1],d[2],'g.')
d = data_new.loc[data_new.label==4.0,:]
ax.scatter(d[0],d[1],d[2],'c.')


ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
plt.show()

