import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets._samples_generator import make_classification
# X为样本特征，y为样本类别输出， 共200个样本，每个样本2个特征，输出有3个类别，没有冗余特征，每个类别一个簇

# Get Data
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

data_raw = pd.read_csv("CW_Data.csv")

# 删去空值行
data_raw = data_raw.dropna()

# 去重复
data_raw = data_raw.drop_duplicates(subset=['Q1', "Q2", "Q3", "Q4", "Q5"], ignore_index=True, keep=False)

# 去ID
data_raw = data_raw[['Q1', "Q2", "Q3", "Q4", "Q5", "Programme"]]

# 去0类: 440
data_raw = data_raw[~data_raw['Programme'].isin([0])].reset_index()[['Q1', "Q2", "Q3", "Q4", "Q5", "Programme"]]

# 分离特征和标签
data_x = data_raw[['Q1', "Q2", "Q3", "Q4", "Q5"]]
data_y = data_raw[['Programme']]
print(data_x, "\n", data_y)

# MinMaxScaler
mm = MinMaxScaler()
data_x = mm.fit_transform(data_x)
print("MinMax:\n", data_x)

# T-SNE

tsne = TSNE(n_components=2, random_state=22)
data_tsne = pd.DataFrame(tsne.fit_transform(data_x))
print(data_tsne)

X = np.array(data_tsne)
y = np.array(data_y)

print(X)
print(y)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
plt.show()

knn = KNeighborsClassifier(n_neighbors=6, weights='distance')
knn.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max,0.2))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# Create color maps
# 给不同区域赋以颜色
cmap_light = ListedColormap(['#FFAFCC', '#00B4D8', '#57cc99', "#a663cc"])
# 给不同属性的点赋以颜色
cmap_bold = ListedColormap(['#FFC8DD', '#90E0EF', '#80ed99', '#b298dc'])

# 将预测的结果在平面坐标中画出其类别区域
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# 也画出所有的训练集数据
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.title("KNN Visual comparison")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
