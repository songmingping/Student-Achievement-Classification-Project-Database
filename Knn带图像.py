import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets._samples_generator import make_classification
# X为样本特征，y为样本类别输出， 共200个样本，每个样本2个特征，输出有3个类别，没有冗余特征，每个类别一个簇

data = pd.read_csv("./CW_deleterror.csv",sep=',',header=0)
data1 = data.iloc[:,:5]
print(data1)
transfer= PCA(n_components=2)
data_new=transfer.fit_transform(data1)
print(data_new)
data_new = pd.DataFrame(data_new)

X=np.array(data_new)
y=np.array(data.iloc[:,5])
print(X)

#之所以生成2个特征值是因为需要在二维平面上可视化展示预测结果，所以只能是2个，3个都不行
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
plt.show() #根据随机生成样本不同，图形也不同

clf = neighbors.KNeighborsClassifier(n_neighbors = 6 , weights='distance')
clf.fit(X, y)  #用KNN来拟合模型，我们选择K=15，权重为距离远近
h = .03  #网格中的步长
#确认训练集的边界
#生成随机数据来做测试集，然后作预测
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h)) #生成网格型二维数据对
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF','#FFBBBB','#FFBBFF']) #给不同区域赋以颜色
cmap_bold = ListedColormap(['#FF0000', '#003300', '#0000FF','#00EE00','#EE0000'])#给不同属性的点赋以颜色
#将预测的结果在平面坐标中画出其类别区域
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# 也画出所有的训练集数据
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()




# 画图
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Create color maps
# 给不同区域赋以颜色
cmap_light = ListedColormap(['#FFAFCC', '#00B4D8', '#57cc99', "#a663cc"])
# 给不同属性的点赋以颜色
cmap_bold = ListedColormap(['#FFC8DD', '#90E0EF', '#80ed99', '#b298dc'])
# 将预测的结果在平面坐标中画出其类别区域
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, alpha=0.4, cmap=cmap_light)
# 也画出所有的训练集数据
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.title("title")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()