'''
支持向量机即 Support Vector Machine，简称 SVM 。SVM模型的主要思想是在样本特征空间上找到最佳的分离超平面（二维是线）
使得训练集上正负样本间隔最大，这个约束使得在感知机的基础上保证可以找到一个最好的分割分离超平面（也就是说感知机会有多个解）。
SVM是用来解决二分类问题的有监督学习算法，在引入了核方法之后SVM也可以用来解决非线性问题。
'''


import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import csv
from numpy import nan
from sklearn.datasets._base import Bunch
from sklearn.model_selection import learning_curve,validation_curve
import matplotlib.pyplot as plt



# 1.读取数据集


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
x_train,x_test,y_train,y_test = train_test_split(real_data.data,real_data.target,test_size=0.6)

# 3.训练svm分类器
classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovr')  # ovr:一对多策略
classifier.fit(x_train,y_train)  # ravel函数在降维时默认是行序优先

# 4.计算svc分类器的准确率
print("训练集：", classifier.score(x_train, y_train))
print("测试集：", classifier.score(x_test, y_test))


# 查看决策函数
print('train_decision_function:\n', classifier.decision_function(x_train))
print('predict_result:\n', classifier.predict(x_train))


y_predict=classifier.predict(x_test)
print(y_predict)
print("直接比对真实值和预测值:\n",y_test==y_predict)

score=classifier.score(x_test,y_test)
print("准确率为：\n",score)

#Learning curve 检视过拟合
train_sizes, train_loss, test_loss = learning_curve(
    svm.SVC(gamma=0.001), real_data.data, real_data.target, cv=10, scoring='neg_mean_squared_error',
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1])

#平均每一轮所得到的平均方差(共5轮，分别为样本10%、25%、50%、75%、100%)
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(train_sizes, train_loss_mean, 'o-', color="r",
         label="Training")
plt.plot(train_sizes, test_loss_mean, 'o-', color="g",
        label="Cross-validation")

plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()


param_range = np.logspace(-6,-2.3,5)

#使用validation_curve快速找出参数对模型的影响
train_loss, test_loss = validation_curve(
    svm.SVC(), x_train, y_train, param_name='gamma', param_range=param_range, cv=10, scoring='neg_mean_squared_error')

#平均每一轮的平均方差
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

#可视化图形
plt.plot(param_range, train_loss_mean, 'o-', color="r",
         label="Training")
plt.plot(param_range, test_loss_mean, 'o-', color="g",
        label="Cross-validation")

plt.xlabel("gamma")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
