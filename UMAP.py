import umap
import matplotlib.pyplot as plt
from sklearn import decomposition,manifold
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


data = pd.read_csv("./CW_deleterror.csv",sep=',',header=0)
data1=data.iloc[:,:5]



umap_data = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=3).fit_transform(data1)
plt.figure(figsize=(12,8))
plt.title('Decomposition using UMAP')
plt.scatter(umap_data[:,0], umap_data[:,1])
plt.scatter(umap_data[:,1], umap_data[:,2])
plt.scatter(umap_data[:,2], umap_data[:,0])
plt.show()

Fit = pd.DataFrame(umap_data)
Fit['label'] = data.Programme

x = Fit.iloc[:, 0]
y = Fit.iloc[:, 1]
z = Fit.iloc[:, 2]

fig = plt.figure()
ax = Axes3D(fig)
d = Fit.loc[Fit.label==0.0,:]
ax.scatter(d[0],d[1],d[2],'r.')
d = Fit.loc[Fit.label==1.0,:]
ax.scatter(d[0],d[1],d[2],'y.')
d = Fit.loc[Fit.label==2.0,:]
ax.scatter(d[0],d[1],d[2],'b.')
d = Fit.loc[Fit.label==3.0,:]
ax.scatter(d[0],d[1],d[2],'p.')
d = Fit.loc[Fit.label==4.0,:]
ax.scatter(d[0],d[1],d[2],'c.')


ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
plt.show()

umap_data = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2).fit_transform(data1)
print(umap_data)


Fit = pd.DataFrame(umap_data)
Fit['label'] = data.Programme



d = Fit.loc[Fit.label==0.0,:]
plt.plot(d[0],d[1],'r.')
d = Fit.loc[Fit.label==1.0,:]
plt.plot(d[0],d[1],'y.')
d = Fit.loc[Fit.label==2.0,:]
plt.plot(d[0],d[1],'b.')
d = Fit.loc[Fit.label==3.0,:]
plt.plot(d[0],d[1],'g.')
d = Fit.loc[Fit.label==4.0,:]
plt.plot(d[0],d[1],'c.')
plt.show()