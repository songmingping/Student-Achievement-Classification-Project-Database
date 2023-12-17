import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sklearn.decomposition as sk_decomposition

data = pd.read_csv("./CW_deleterror.csv",sep=',',header=0)
data1 = data.iloc[:,:5]
print(data1)
pca=sk_decomposition.PCA(n_components=0.8,whiten=False,svd_solver='auto')
#n_components是降维后的特征数（维度）
#whiten:判断是否对降维后的每个数据进行归一化，让方差都为1
#svd_solver奇异值分解的方法，包括{'auto','full','arpack','randomized'}
reduced_X=pca.fit_transform(data1)
# reduced_X=pca.transform(data1)#降维后的数据
print(reduced_X)
print(pca.explained_variance_ratio_)#降维后的各主成分的方差值占总方差值的比例
print(pca.explained_variance_)#降维后的各主成分的方差值
print(pca.n_components)#降维后的特征数
n_cluster=range(1,10)
kmeans=[KMeans(n_clusters=i).fit(reduced_X)for i in n_cluster]
scores=[kmeans[i].score(reduced_X)for i in range(len(kmeans))]

plt.plot(n_cluster,scores)

plt.xlabel('n_cluster')
plt.ylabel('score')
plt.title('Elbow plot')
plt.show()


transfer= PCA(n_components=2)
data_new=transfer.fit_transform(data1)
print(data_new)
data_new = pd.DataFrame(data_new)
scaler=StandardScaler()
np_scaled=scaler.fit_transform(data_new)
data=pd.DataFrame(np_scaled)

kmeans=[KMeans(n_clusters=i).fit(data)for i in n_cluster]

#计算每个数据点到其聚类中心的距离
def getDistanceByPoint(data,model):
    distance=pd.Series()
    for i in range(0,len(data)):
        Xa=np.array(data.loc[i])
        Xb=model.cluster_centers_[model.labels_[i]]
        distance.at [i]=np.linalg.norm(Xa-Xb)
    return distance
#设置异常值比例
outliers_fraction =0.01
#得到每个点到取聚类中心的距离，我们设置了6个聚类中心，kmeans[6]表示有6个聚类中心的模型
distance=getDistanceByPoint(data,kmeans[6])
#根据异常值比例outliers fraction计算异常值的数量
number_of_outliers =int(outliers_fraction*len(distance))
#设定异常值的闻值
threshold=distance.nlargest(number_of_outliers).min()
#根据阈值来判断是否为异常值
data['anomaly1']=(distance >=threshold).astype(int)
#数据可视化
fig,ax=plt.subplots(figsize=(10,6))
colors ={0:'blue',1:'red'}
print(data)
ax.scatter(data[0],data[1],c=data["anomaly1"].apply(lambda x:colors[x]))
plt.xlabel('principal feature1')
plt.ylabel('principal feature2')
plt.show()

df = data.sort_values(0)
df[0] = df.iloc[:,0].astype(np.int64)
print(df)
fig, ax = plt.subplots(figsize=(12, 6))

a = df.loc[df['anomaly1'] == 1, [0, 1]]  # anomaly

ax.plot(df[0], df[1], color='blue', label='正常值')
ax.scatter(a[0], a[1], color='red', label='异常值')

plt.legend()
plt.show()







