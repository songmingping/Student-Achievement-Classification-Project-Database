'''
k临近算法
如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别。

交叉验证：将拿到的训练数据，分为训练和验证集。将数据分成5份，其中一份作为验证集。然后经过5次(组)的测试，每次都更换不同的验证集。即得到5组模型的结果，
取平均值作为最终结果。又称5折交叉验证。

超参数搜索-网格搜索(Grid Search)
通常情况下，有很多参数是需要手动指定的（如k-近邻算法中的K值），这种叫超参数。但是手动过程繁杂，所以需要对模型预设几种超参数组合。
每组超参数都采用交叉验证来进行评估。最后选出最优参数组合建立模型。

sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,algorithm='auto')
n_neighbors：int,可选（默认= 5），k_neighbors查询默认使用的邻居数
algorithm：{‘auto’，‘ball_tree’，‘kd_tree’，‘brute’}，可选用于计算最近邻居的算法：‘ball_tree’将会使用 BallTree，‘kd_tree’将使用 KDTree。
‘auto’将尝试根据传递给fit方法的值来决定最合适的算法。 (不同实现方式影响效率)
'''
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import csv
from numpy import nan
from sklearn.datasets._base import Bunch
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import  GridSearchCV
import sklearn.model_selection as skm
import sklearn.neighbors as skn
import joblib


#导入并处理数据转换成bunch格式文件
csvFile = open("finaldata.csv")
csv_data = csv.reader(csvFile)
cancer=np.array([i for i in csv_data])
attribute_names=cancer[0,:5]
da=cancer[1:,:5]
data=[]
for i in da:
    temp=[]
    for j in i:
        if j=='?':
            temp.append(nan)
        else:
            temp.append(float(j))
    data.append(temp)
attribute_names=cancer[0,:4]
target=[]
for i in cancer[1:,5]:
    if i=='0':
        target.append(0)
    if i=='1':
        target.append(1)
    if i=='2':
        target.append(2)
    if i=='3':
        target.append(3)
    if i=='4':
        target.append(4)
target_names=['0','1','2','3','4']
real_data = Bunch(data=data, target=target, feature_names= attribute_names, target_names = target_names)
x_train,x_test,y_train,y_test  =  train_test_split(real_data.data,real_data.target,test_size=0.6,random_state=22)

#标准化
transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)
x_train = pd.DataFrame(x_train)
print(x_train.values.shape)

#网格搜索与交叉验证
param_dict={"n_neighbors":[i for i in range(1,10)]}
estimator=KNeighborsClassifier(algorithm='kd_tree')
joblib.dump(estimator,"k临近算法.pkl")
# estimator=joblib.load("k临近算法.pkl")
estimator1=GridSearchCV(estimator,param_grid=param_dict,cv=5)
estimator1.fit(x_train,y_train)


y_predict=estimator1.predict(x_test)
print(y_predict)
print("直接比对真实值和预测值:\n",y_test==y_predict)
print(y_predict==0)
score=estimator1.score(x_test,y_test)
print("准确率为：\n",score)
print("最佳参数：\n",estimator1.best_params_)
print("最佳结果：\n",estimator1.best_score_)
print("最佳估计器：\n",estimator1.best_estimator_)
print("交叉验证结果：\n",estimator1.cv_results_)


#交叉验证
from sklearn.model_selection import cross_val_score # K折交叉验证模块

#使用K折交叉验证模块
scores = cross_val_score(estimator, x_train, y_train, cv=5, scoring='accuracy')
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator , X = x_train, y = y_train, cv = 5)

accuracies.mean()

accuracies.std()
#将5次的预测准确率打印出
print(scores)
#将5次的预测准确平均率打印出
print(scores.mean())
#建立测试参数集
k_range = range(1, 31)

k_scores = []

#藉由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())


#可视化数据
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


#平均方差
from sklearn.metrics import mean_squared_error
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    loss = -cross_val_score(knn, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
    k_scores.append(loss.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated MSE')
plt.show()









#error图像
kRange=range(1,200)
kError=[]
for k in kRange:
    knn=skn.KNeighborsClassifier(n_neighbors=k)
    scores=skm.cross_val_score(knn,x_train,y_train,scoring='accuracy')
    kError.append(1-scores.mean())

plt.plot(kRange,kError)
plt.xlabel('Value of k')
plt.ylabel('error')
plt.show()



