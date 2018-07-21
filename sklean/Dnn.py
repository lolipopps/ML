# -*- coding: utf8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import GridSearchCV
# from pybrain.structure import *
from sklearn.neural_network import MLPClassifier
# from pybrain.supervised.trainers import BackpropTrainer
from sklearn.model_selection import cross_val_score
MLPClassifier(alpha=1),

#    加载数据集
def load_data():
    train = pd.read_csv(r'./data/features.dat', sep=',')
    label = pd.read_csv(r'./data/labels.dat', sep=',')
    #    train = StandardScaler().fit_transform(train)    #标准数据预处理
    X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=.2,
                                                        random_state=0)  # 其中test_size是按照8:2，random是随机取数据
    return train, label, X_train, X_test, y_train, y_test


train, label, X_train, X_test, y_train, y_test = load_data();
classes = np.unique(y_train)
### 会自动根据标签读取个数
mlp = MLPClassifier(hidden_layer_sizes=(80,30), max_iter=200, alpha=1e-5,
                    solver='adam', verbose=100, tol=1e-4, random_state=1,
                    learning_rate_init=0.001).fit(X_train, y_train)

"""
1. hidden_layer_sizes :元祖格式，长度=n_layers-2, 默认(100，） 神经网络层数
2. activation :{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, 默认‘relu   identity   f(x) = x 
4. solver： {‘lbfgs’, ‘sgd’, ‘adam’}, 默认 ‘adam’，用来优化权重 默认solver ‘adam’在相对较大的数据集上效果比较好（几千个样本或者更多），对小数据集来说，lbfgs收敛更快效果也更好。 
5. alpha :float,可选的，默认0.0001,正则化项参数 
6. batch_size : int , 可选的，默认‘auto’,随机优化的minibatches的大小，如果solver是‘lbfgs’，分类器将不使用minibatch，当设置成‘auto’，batch_size=min(200,n_samples) 
7. learning_rate :{‘constant’，‘invscaling’, ‘adaptive’},默认‘constant’，用于权重更新，只有当solver为’sgd‘时使用 
- ‘constant’: 有‘learning_rate_init’给定的恒定学习率 
- ‘incscaling’：随着时间t使用’power_t’的逆标度指数不断降低学习率learning_rate_ ，effective_learning_rate = learning_rate_init / pow(t, power_t) 
- ‘adaptive’：只要训练损耗在下降，就保持学习率为’learning_rate_init’不变，当连续两次不能降低训练损耗或验证分数停止升高至少tol时，将当前学习率除以5. 
8. max_iter: int，可选，默认200，最大迭代次数。 
9. random_state:int 或RandomState，可选，默认None，随机数生成器的状态或种子。 
10. shuffle: bool，可选，默认True,只有当solver=’sgd’或者‘adam’时使用，判断是否在每次迭代时对样本进行清洗。 
11. tol：float, 可选，默认1e-4，优化的容忍度 
12. learning_rate_int:double,可选，默认0.001，初始学习率，控制更新权重的补偿，只有当solver=’sgd’ 或’adam’时使用。 
13. power_t: double, optional, default 0.5，只有solver=’sgd’时使用，是逆扩展学习率的指数.当learning_rate=’invscaling’，用来更新有效学习率。 
14. verbose : bool, optional, default False,是否将过程打印到stdout 
15. warm_start : bool, optional, default False,当设置成True，使用之前的解决方法作为初始拟合，否则释放之前的解决方法。 
16. momentum : float, default 0.9,Momentum(动量） for gradient descent update. Should be between 0 and 1. Only used when solver=’sgd’. 
17. nesterovs_momentum : boolean, default True, Whether to use Nesterov’s momentum. Only used when solver=’sgd’ and momentum > 0. 
18. early_stopping : bool, default False,Only effective when solver=’sgd’ or ‘adam’,判断当验证效果不再改善的时候是否终止训练，当为True时，自动选出10%的训练数据用于验证并在两步连续爹迭代改善低于tol时终止训练。 
19. validation_fraction : float, optional, default 0.1,用作早期停止验证的预留训练数据集的比例，早0-1之间，只当early_stopping=True有用 
20. beta_1 : float, optional, default 0.9，Only used when solver=’adam’，估计一阶矩向量的指数衰减速率，[0,1)之间 
21. beta_2 : float, optional, default 0.999,Only used when solver=’adam’估计二阶矩向量的指数衰减速率[0,1)之间 
22. epsilon : float, optional, default 1e-8,Only used when solver=’adam’数值稳定值。 
属性说明： 
- classes_:每个输出的类标签 
- loss_:损失函数计算出来的当前损失值 
- coefs_:列表中的第i个元素表示i层的权重矩阵 
- intercepts_:列表中第i个元素代表i+1层的偏差向量 
- n_iter_ ：迭代次数 
- n_layers_:层数 
- n_outputs_:输出的个数 
- out_activation_:输出激活函数的名称。 
方法说明： 
- fit(X,y):拟合 
- get_params([deep]):获取参数 
- predict(X):使用MLP进行预测 
- predic_log_proba(X):返回对数概率估计 
- predic_proba(X)：概率估计 
- score(X,y[,sample_weight]):返回给定测试数据和标签上的平均准确度 
-set_params(**params):设置参数。
"""
def getBestPara():
    grid = GridSearchCV(MLPClassifier(), param_grid={"alpha":[1e-4, 1e-3, 1e-5], "solver": ['sgd', 'adam'],
                                                   "learning_rate_init":[0.1,0.01,0.001]}, cv=3)
    grid.fit(X_train, y_train)
    print("The best parameters are %s with a score of %0.2f"% (grid.best_params_, grid.best_score_))

scores = cross_val_score(mlp, X_test, y_test, cv=5) ## cross_val_score 指定参数来设定评测标准 默认使用KFold 或StratifiedKFold 进行数据集打乱，
print(scores)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))
print(mlp.n_layers_)
print (mlp.n_iter_)
print (mlp.loss_)
print(mlp.n_outputs_)
print(mlp.classes_)
print (mlp.out_activation_)