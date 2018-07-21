
# -*- coding: utf8 -*-
# -*- coding: utf8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import cross_validation, metrics  ## 计算 AUC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold  ## 可以不用
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer
"""
clf = svm.SVC()
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
NuSvc   Similar to SVC but uses a parameter to control the number of support vectors.

（1）C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0； 
（2）kernel：参数选择有RBF, Linear, Poly, Sigmoid, 默认的是"RBF"; 
（3）degree：if you choose 'Poly' in param 2, this is effective, degree决定了多项式的最高次幂； 
（4）gamma：核函数的系数('Poly', 'RBF' and 'Sigmoid'), 默认是gamma = 1 / n_features; 
（5）coef0：核函数中的独立项，'RBF' and 'Poly'有效； 
（6）probablity: 可能性估计是否使用(true or false)； 
（7）shrinking：是否进行启发式； 
（8）tol（default = 1e - 3）: svm结束标准的精度; 
（9）cache_size: 制定训练所需要的内存（以MB为单位）； 
（10）class_weight: 每个类所占据的权重，不同的类设置不同的惩罚参数C, 缺省的话自适应； 
（11）verbose: 跟多线程有关，不大明白啥意思具体； 
（12）max_iter: 最大迭代次数，default = 1， if max_iter = -1, no limited; 
（13）decision_function_shape ： ‘ovo’ 一对一, ‘ovr’ 多对多  or None 无, default=None 
（14）random_state ：用于概率估计的数据重排时的伪随机数生成器的种子。 

，通过关键字sample_weight。类似于class_weight这些，将C第i个示例的参数设置为。C * sample_weight[i]

NuSVC(nu=0.5, kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True,
 probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False,
 max_iter=-1, decision_function_shape=’ovr’, random_state=None)[source]


 数据不平衡问题
 可以使用 SGDClassifier(loss="hinge")代替SVC(kernel="linear")

 支持分类的支持向量机可以推广到解决回归问题，这种方法称为支持向量回归
 支持向量回归SVR，nusvr和linearsvr。linearsvr提供了比SVR更快实施但只考虑线性核函数，而nusvr实现比SVR和linearsvr略有不同。

 SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',  
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)  


"""


def Manclass():
    X = [[0], [1], [2], [3]]
    Y = [0, 1, 2, 3]
    """
        SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    """
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X, Y)


'''
    dec = clf.decision_function([[1]])
    >> > dec.shape[1]  # 4 classes: 4*3/2 = 6
    6
    >> > clf.decision_function_shape = "ovr"
    >> > dec = clf.decision_function([[1]])
    >> > dec.shape[1]  # 4 classes
    4
'''


## 保存模型
def saveModel(filename, model):  ## 交叉验证模型保存
    joblib.dump(model, filename)


def loadModel(filename):
    clf = joblib.load(filename)
    return clf


def load_data():  ## 加载数据集
    train = pd.read_csv('./data/BiHuaFeature.csv', sep=',',header='infer')
    shapes = train.shape;
    print(train.columns)
    label = train['class']
    train = train.drop('class', 1)
    train = Imputer().fit_transform(train)
    print(label.shape,train.shape)
    train = StandardScaler().fit_transform(train)
    X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=.2, random_state=0)  ## 这一行可以不用
    return train, label, X_train, X_test, y_train, y_test


train, label, X_train, X_test, y_train, y_test = load_data();


### 调参数 9种组合看哪个好 4路交叉验证 采用的是高斯核函数 找出一个好的参数
def getBestPara():
    grid = GridSearchCV(svm.SVC(), param_grid={"C": [0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, cv=4)
    grid.fit(X_train, y_train)
    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))


## 训练模型 采用的是线性核函数
def Train():
    clf = svm.SVC(C=10.0, gamma=0.01, kernel='rbf', ).fit(X_train, y_train)  ## 默认使用 线性核函数  训练模型 使用  X_train, y_train 数据集
    scores = cross_val_score(clf, X_test, y_test,
                             cv=5)  ## cross_val_score 指定参数来设定评测标准 默认使用KFold 或StratifiedKFold 进行数据集打乱，
    print(scores)
    ## 另一种方式实现交叉验证
    # cv = ShuffleSplit(n_splits=3, test_size=.3, random_state=0)  ## 对交叉验证方式进行指定，如验证次数，训练集测试集划分比例等
    # crss_score = cross_val_score(clf, X_test, y_test, cv=cv)
    saveModel('./data/modelSvm.m', clf)  ## 保存模型


## 对模型进行交叉验证  最后输出准确率
def Predict(clf):
    if clf == None:
        clf = loadModel('./data/modelSvm.m')
    scores = cross_val_score(clf, X_test, y_test, cv=5)
    predict_prob_y = clf.predict(X_test)  # 基于SVM对验证集做出预测，prodict_prob_y 为预测的概率
    test_auc = metrics.roc_auc_score(y_test, predict_prob_y)  # 验证集上的auc值
    print(test_auc)
    for i in range(1, 10):
        print(clf.predict(X_test[i - 1:i]), print(y_test[i - 1:i]))


def main(para):
    if para == 1:
        Train();
        Predict(clf=None);
    else:
        print("---------")
        getBestPara()


if __name__ == "__main__":
    main(1);




