import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import tree
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE, Isomap
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import warnings

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


# Get Data
data_raw = pd.read_csv("CW_Data.csv")
# 数据检查
# print(data_raw.head())
# 使用info()查看缺失值:info()这个方法能够统计每个属性下非空值的数量，总个数以及数据量的大小
# print(data_raw.info())
# 总共515行，有514行非空，说明有一行全空

# 使用apply()统计每一个属性的缺失率:百分比显示缺失率更加直观，对于缺失率高的属性，可以考虑删除
# print(data.T.apply(lambda x: '{}%'.format(round(100*sum(x.isnull())/len(x), 2))))

# 删去空值行
data_raw = data_raw.dropna()

# 去重复
data_raw = data_raw.drop_duplicates(subset=['Q1', "Q2", "Q3", "Q4", "Q5"], ignore_index=True, keep=False)

# 去ID
data_raw = data_raw[['Q1', "Q2", "Q3", "Q4", "Q5", "Programme"]]


# 各专业数据汇总
def count(data_in):
    P1 = (data_in.loc[data_in.Programme == 0.0, :]).iloc[:, :-1]
    P2 = (data_in.loc[data_in.Programme == 1.0, :]).iloc[:, :-1]
    P3 = (data_in.loc[data_in.Programme == 2.0, :]).iloc[:, :-1]
    P4 = (data_in.loc[data_in.Programme == 3.0, :]).iloc[:, :-1]
    P5 = (data_in.loc[data_in.Programme == 4.0, :]).iloc[:, :-1]

# 各专业人数
    PL1 = len(P1)
    PL2 = len(P2)
    PL3 = len(P3)
    PL4 = len(P4)
    PL5 = len(P5)
# 各专业人数柱状图
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.figure()
    sns.set(style='whitegrid')
    plt.title('Person Number of Each Programme')
    plt.bar(['P0', 'P1', 'P2', 'P3', 'P4'], [PL1, PL2, PL3, PL4, PL5])
    for a, b in zip(['P0', 'P1', 'P2', 'P3', 'P4'], [PL1, PL2, PL3, PL4, PL5]):
        plt.text(a, b+1, b, ha='center', va='bottom')
    plt.show()


# count(data_raw)

# 去0类: 440
data_raw = data_raw[~data_raw['Programme'].isin([0])].reset_index()[['Q1', "Q2", "Q3", "Q4", "Q5", "Programme"]]
data_raw.to_csv(r'D:/PROJECT/Python/CW2/1.csv', index=False)
# 分离特征和标签
data_x = data_raw[['Q1', "Q2", "Q3", "Q4", "Q5"]]
data_y = data_raw[['Programme']]


# Boxplot for each feature
def boxPlot(data_in):
    data_in.plot.box(title="Box-plot")
    plt.grid(linestyle="--", alpha=0.3)
    outlier = data_in.boxplot(return_type='dict')
    y = outlier['fliers'][2].get_ydata()
    plt.show()


# boxPlot(data_x)


# Pair plot
def pairPlot(data):
    plt.rcParams['figure.figsize'] = (15, 15)
    sns.pairplot(data, hue='Programme', palette="husl")
    plt.title("Pair Plot")
    plt.show()


# pairPlot(data_raw)


# 可视化分布情况
def distribution(data_in):
    plt.rcParams['figure.figsize'] = (12, 15)
    plt.subplot(3, 2, 1)
    sns.set(style='whitegrid')
    sns.distplot(data_in['Q1'])
    plt.title('Distribution of Q1')
    plt.xlabel('Range of Q1')
    plt.ylabel('Count')


    plt.subplot(3, 2, 2)
    sns.set(style='whitegrid')
    sns.distplot(data_in['Q2'])
    plt.title('Distribution of Q2')
    plt.xlabel('Range of Q2')
    plt.ylabel('Count')


    plt.subplot(3, 2, 3)
    sns.set(style='whitegrid')
    sns.distplot(data_in['Q3'])
    plt.title('Distribution of Q3')
    plt.xlabel('Range of Q3')
    plt.ylabel('Count')


    plt.subplot(3, 2, 4)
    sns.set(style='whitegrid')
    sns.distplot(data_in['Q4'])
    plt.title('Distribution of Q4')
    plt.xlabel('Range of Q4')
    plt.ylabel('Count')


    plt.subplot(3, 2, 5)
    sns.set(style='whitegrid')
    sns.distplot(data_in['Q5'])
    plt.title('Distribution of Q5')
    plt.xlabel('Range of Q5')
    plt.ylabel('Count')
    plt.show()


# distribution(data_x)

# 数据相关性
def pearsonCorr(data_in):
    plt.rcParams['figure.figsize'] = (15, 8)
    sns.heatmap(data_in.corr(), cmap="Wistia", annot=True)
    plt.title('Pearson correlation coefficient')
    plt.show()


# pearsonCorr(data_x)

# MinMaxScaler
mm = MinMaxScaler()
data_x = mm.fit_transform(data_x)


# Regression
# Visualize
def draw(inputData, label, title):
    inputData['Programme'] = label.Programme
    plt.rcParams['figure.figsize'] = (15, 8)
    plt.figure()
    sns.set(style='whitegrid')
    plt.title(title)
    plt.scatter((inputData.loc[inputData.Programme == 1.0, :])[0], (inputData.loc[inputData.Programme == 1.0, :])[1],
                c='pink', label='Programme 1')
    plt.scatter((inputData.loc[inputData.Programme == 2.0, :])[0], (inputData.loc[inputData.Programme == 2.0, :])[1],
                c='green', label='Programme 2')
    plt.scatter((inputData.loc[inputData.Programme == 3.0, :])[0], (inputData.loc[inputData.Programme == 3.0, :])[1],
                c='magenta', label='Programme 3')
    plt.scatter((inputData.loc[inputData.Programme == 4.0, :])[0], (inputData.loc[inputData.Programme == 4.0, :])[1],
                c='orange', label='Programme 4')
    plt.legend(loc="upper right")
    plt.show()


# PCA
def pcaRegression(x, n):
    pca = PCA(n_components=n, random_state=7)
    data_out = pd.DataFrame(pca.fit_transform(x))
    return data_out


# NMF
def nmfRegression(x, n):
    nmf = NMF(n_components=n, solver="mu", max_iter=200, random_state=7, init='nndsvda')
    data_out = pd.DataFrame(nmf.fit_transform(x))
    return data_out


# T-SNE
def tsneRegression(x, n):
    tsne = TSNE(n_components=n, learning_rate=777, random_state=7)
    data_out = pd.DataFrame(tsne.fit_transform(x))
    return data_out


# ISOMAP
def isomapRegression(x, n):
    isomap = Isomap(neighbors_algorithm='kd_tree', n_components=n)
    data_out = pd.DataFrame(isomap.fit_transform(x))
    return data_out


data_pca = pcaRegression(data_x, 2)
data_nmf = nmfRegression(data_x, 2)
data_tsne = tsneRegression(data_x, 2)
data_isomap = isomapRegression(data_x,2)

# draw(data_isomap, data_y, "Isomap")

# draw(data_tsne, data_y, "T-SNE")

# Using tsne
data_x = data_tsne
# Data split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25, random_state=7)


def compareVisually(x_tr, y_tr, clf, title):
    # Compare Visually
    data_dtx = np.array(x_tr)
    data_dty = np.array(y_tr)
    # 拟合模型
    clf.fit(x_tr, y_tr)

    # 画图
    x_min, x_max = data_dtx[:, 0].min() - 1, data_dtx[:, 0].max() + 1
    y_min, y_max = data_dtx[:, 1].min() - 1, data_dtx[:, 1].max() + 1
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
    plt.scatter(data_dtx[:, 0], data_dtx[:, 1], c=data_dty, cmap=cmap_bold)
    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


# KNN
def KNNK_Fold(x, y, start, end):
    """

    :param x: features
    :param y: labels
    :param start: k's range start
    :param end: k's range end
    :return:
    """
    k_Range = range(start, end)
    k_Error = []
    for k in k_Range:
        clf = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree', weights="uniform")
        scores = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
        k_Error.append(1 - scores.mean())
    # Draw the pic of K
    plt.plot(k_Range, k_Error)
    plt.xlabel('Value of K in KNN')
    plt.ylabel('Error')
    plt.show()
    minK = min(k_Error)
    for k in k_Range:
        if k_Error[k] == minK:
            return k+start


# GridSearchCV
def KNNGridSearchCV(x_tr, x_te, y_tr, y_te):
    clf = KNeighborsClassifier()
    param = {
        "n_neighbors": [i for i in range(1, 30)],
        "weights": ["distance", "uniform"],
        "algorithm": ['ball_tree', 'kd_tree', 'brute'],
    }
    gc = GridSearchCV(clf, param_grid=param, cv=5)
    gc.fit(x_tr, y_tr)
    print("Best Param:", gc.best_params_)
    print("KNN:", gc.score(x_te, y_te))


def K_N_N():
    K = KNNK_Fold(x_train, y_train, 1, 50)
    print(K)
    knn = KNeighborsClassifier(n_neighbors=18, algorithm='ball_tree', weights="uniform")
    # 调用fit()
    knn.fit(x_train, y_train)
    # 预测测试数据集，得出准确率
    y_pred = knn.predict(x_test)
    print("预测测试集类别：", y_pred)
    print("准确率为：", knn.score(x_test, y_test))
    # KNN Confusion Matrix
    C = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4])
    df = pd.DataFrame(C, index=[1, 2, 3, 4], columns=[1, 2, 3, 4])
    plt.title("KNN Confusion Matrix")
    plt.ylabel("Predict")
    plt.xlabel("True")
    sns.heatmap(df, annot=True, cmap='Oranges')
    plt.show()
    # Evaluation
    print(classification_report(y_test, y_pred))
    # Cv
    compareVisually(x_train, y_train, knn, "KNN Visual Compare")


# KNNGridSearchCV(x_train, x_test, y_train, y_test)
K_N_N()


# Decision Tree
def decisionTreeGridSearchCV(x_tr, x_te, y_tr, y_te):
    dt = DecisionTreeClassifier()
    param = {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [i for i in range(1, 25)],
        'random_state': [i for i in range(1, 25)],
        'max_leaf_nodes': [i for i in range(1, 25)]
    }
    gc = GridSearchCV(dt, param_grid=param, cv=5)
    gc.fit(x_tr, y_tr)
    print("Best Param:", gc.best_params_)
    print(gc.score(x_te, y_te))


# decisionTreeGridSearchCV(x_train, x_test, y_train, y_test)


def decisionTree():
    dt = DecisionTreeClassifier(criterion='gini', max_depth=7, splitter='random', max_leaf_nodes=14, random_state=3)
    # 调用fit()
    dt.fit(x_train, y_train)
    # 预测测试数据集，得出准确率
    y_pred = dt.predict(x_test)
    print("预测测试集类别：", y_pred)
    print("准确率为：", dt.score(x_test, y_test))

    # KNN Confusion Matrix
    C = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4])
    df = pd.DataFrame(C, index=[1, 2, 3, 4], columns=[1, 2, 3, 4])
    plt.title("Decision Tree Confusion Matrix")
    plt.ylabel("Predict")
    plt.xlabel("True")
    sns.heatmap(df, annot=True, cmap='Oranges')
    plt.show()
    # Evaluation
    print(classification_report(y_test, y_pred))
    # Visualize
    tree.plot_tree(dt, filled=True)
    plt.show()
    compareVisually(x_train, y_train, dt, "Decision Tree Visual Compare")


decisionTree()





