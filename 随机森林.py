'''
集成学习方法
集成学习通过建立几个模型组合的来解决单一预测问题。它的工作原理是生成多个分类器/模型，各自独立地学习和作出预测。这些预测最后结合成组合预测，
因此优于任何一个单分类的做出预测。

在机器学习中，随机森林是一个包含多个决策树的分类器，并且其输出的类别是由个别树输出的类别的众数而定。

用N来表示训练用例（样本）的个数，M表示特征数目。
1、一次随机选出一个样本，重复N次， （有可能出现重复的样本）
2、随机去选出m个特征, m <<M，建立决策树
采取bootstrap抽样

为什么要随机抽样训练集？　　
如果不进行随机抽样，每棵树的训练集都一样，那么最终训练出的树分类结果也是完全一样的
为什么要有放回地抽样？
如果不是有放回的抽样，那么每棵树的训练样本都是不同的，都是没有交集的，这样每棵树都是“有偏的”，都是绝对“片面的”（当然这样说可能不对），
也就是说每棵树训练出来都是有很大的差异的；而随机森林最后分类取决于多棵树（弱分类器）的投票表决。

class sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion=’gini’, max_depth=None, bootstrap=True,
random_state=None, min_samples_split=2)

随机森林分类器
n_estimators：integer，optional（default = 10）森林里的树木数量120,200,300,500,800,1200
criteria：string，可选（default =“gini”）分割特征的测量方法
max_depth：integer或None，可选（默认=无）树的最大深度 5,8,15,25,30
max_features="auto”,每个决策树的最大特征数量
If "auto", then max_features=sqrt(n_features).
If "sqrt", then max_features=sqrt(n_features) (same as "auto").
If "log2", then max_features=log2(n_features).
If None, then max_features=n_features.
bootstrap：boolean，optional（default = True）是否在构建树时使用放回抽样
min_samples_split:节点划分最少样本数
min_samples_leaf:叶子节点的最小样本数
超参数：n_estimator, max_depth, min_samples_split,min_samples_leaf
'''
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import csv
from numpy import nan
from sklearn.datasets._base import Bunch
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  GridSearchCV
import joblib
import sklearn.model_selection as skm
import sklearn.neighbors as skn


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
x_train,x_test,y_train,y_test  =  train_test_split(real_data.data,real_data.target,test_size=0.3,random_state=20)

transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.fit_transform(x_test)
transfer=MinMaxScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.fit_transform(x_test)

estimator=RandomForestClassifier()
param_dict={"n_estimators":[i for i in range(100,1500,100)],"max_depth":[i for i in range(1,20,1)]}
estimator=GridSearchCV(estimator,param_grid=param_dict,cv=3,scoring='accuracy')
estimator.fit(x_train,y_train)
y_predict=estimator.predict(x_test)
print(y_predict)
print("直接比对真实值和预测值:\n",y_test==y_predict)

score=estimator.score(x_test,y_test)
score_train=estimator.score(x_train,y_train)
print("随机森林训练集准确率为：\n",score_train)
print("测试集准确率为：\n",score)

print("最佳参数：\n",estimator.best_params_)
print("最佳结果：\n",estimator.best_score_)
print("最佳估计器：\n",estimator.best_estimator_)
print("交叉验证结果：\n",estimator.cv_results_)

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
k_range=[i for i in range(50,1500,50)]
k_scores = []
#藉由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率
for k in k_range:
    knn = RandomForestClassifier(n_estimators=k)
    scores = cross_val_score(knn, x_train, y_train, cv=3, scoring='accuracy')
    k_scores.append(scores.mean())


#可视化数据
plt.plot(k_range, k_scores)
plt.xlabel('Value of n_estimators for Random Forest')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


k_range=[i for i in range(1,20,1)]
k_scores = []
#藉由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率
for k in k_range:
    RFC = RandomForestClassifier(max_depth=k)
    scores = cross_val_score(RFC, x_train, y_train, cv=3, scoring='accuracy')
    k_scores.append(scores.mean())


#可视化数据
plt.plot(k_range, k_scores)
plt.xlabel('Value of max_depth for Random Forest')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


