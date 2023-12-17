#等度量映射（MDS变种）
#全局模型
#效果不好（降到1维的代码有问题）
# isomap 最主要的优点就是使用“测地距离”，而不是使用原始的欧几里得距离，这样可以更好的控制数据信息的流失，能够在低维空间中更加全面的将高维空间的数据表现出来。




import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition,manifold
import pandas as pd

def load_data():
    data = pd.read_csv("./CW_deleterror.csv",sep=',',header=0)
    data1 = data.iloc[:, :5]
    return data1,data.Programme

def test_Isomap(*data):
    X,y=data
    for n in [5,4,3,2,1]:
        isomap = manifold.Isomap(n_components=n)
        isomap.fit(X)
        print('reconstruction_error(n_components=%d):%s'%(n,
            isomap.reconstruction_error()))
X,y=load_data()
test_Isomap(X,y)

def plot_Isomap(*data):
    X,y=data
    Ks=[1,5,25,y.size-1]
    fig=plt.figure()
    for i,k in enumerate(Ks):
        isomap=manifold.Isomap(n_components=2,n_neighbors=k)
        X_r=isomap.fit_transform(X)
        ax=fig.add_subplot(2,2,i+1)
        colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),
               (0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2),)
        for label,color in zip(np.unique(y),colors):
            position=y==label
            ax.scatter(X_r[position,0],X_r[position,1],label='target=%d'%label,color=color)
        ax.set_xlabel('X[0]')
        ax.set_ylabel('X[1]')
        ax.legend(loc='best')
        ax.set_title("k=%d"%k)
    plt.suptitle('Isomap')
    plt.tight_layout(pad=2)
    plt.show()
plot_Isomap(X,y)



