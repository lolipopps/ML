# -*- coding: utf8 -*-
"""
所有的线性模型都位于sklearn.linear_model下，包含线性回归、岭回归、lasso、弹性网、最小角回归，
及各自的多任务版本和带交叉验证的版本，还包括一些其他的模型，如感知机(Perceptron)   线性回归通过最小化均方误差来拟合一个线性模型，
属于监督学习，对于给定的数据集X和类标签y，最小化均方误差
"""
from sklearn import linear_model
## LinearRegression
"""
fit_intercept：初始化时的一个参数，默认为True，是否计算b值
normalize：是否需要将样本归一化
n_jobs：并行时的CPU的线程数，如果为-1，则调用所有可用的CPU线程,线性回归实现了回归器的基类RegressorMixin，以及一个抽象类LinearModel：
fit  predict 方法
"""
reg = linear_model.LinearRegression()
reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print(reg.coef_)

### Ridge Regression  岭回归即我们所说的L2正则线性回归
"""
fit_intercept：是否求解b值
tol：判断迭代是否收敛的阈值，当误差小于tol时，停止迭代
random_state：可以传入相同的整数值来使得运算结果可以重现，也可以是RandomState对象
alpha：即正则化系数，必须为一个正数。
solver：求解算法  auto：视数据类型选定相应的算法，svd：通过奇异值分解计算岭回归的系数，针对奇异矩阵有较好的稳定性
cholesky：调用scipy.linalg.solve，得到的是一个闭式解 sparse_cg：使用scipy.sparse.linalg.cg()，即共轭梯度法，是一个迭代算法，比cholesky在大规模数据上表现的更好
lsqr：最小平方QR分解，调用scipy.sparse.linalg.lsqr()  sag：随机平均梯度下降法，注意，如果矩阵是稀疏矩阵并且return_intercept=True那么solver只能为sag，如果不是会给出警告信息并转换成sag

"""
from sklearn import linear_model

reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])

reg = linear_model.Lasso(alpha = 0.1)

## Multil-task lasso 多元回归系数的线性模型

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import MultiTaskLasso, Lasso

rng = np.random.RandomState(42)

# Generate some 2D coefficients with sine waves with random frequency and phase
n_samples, n_features, n_tasks = 100, 30, 40
n_relevant_features = 5
coef = np.zeros((n_tasks, n_features))
times = np.linspace(0, 2 * np.pi, n_tasks)
for k in range(n_relevant_features):
    coef[:, k] = np.sin((1. + rng.randn(1)) * times + 3 * rng.randn(1))

X = rng.randn(n_samples, n_features)
Y = np.dot(X, coef.T) + rng.randn(n_samples, n_tasks)

coef_lasso_ = np.array([Lasso(alpha=0.5).fit(X, y).coef_ for y in Y.T])
coef_multi_task_lasso_ = MultiTaskLasso(alpha=1.).fit(X, Y).coef_

# #############################################################################
# Plot support and time series
fig = plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.spy(coef_lasso_)
plt.xlabel('Feature')
plt.ylabel('Time (or Task)')
plt.text(10, 5, 'Lasso')
plt.subplot(1, 2, 2)
plt.spy(coef_multi_task_lasso_)
plt.xlabel('Feature')
plt.ylabel('Time (or Task)')
plt.text(10, 5, 'MultiTaskLasso')
fig.suptitle('Coefficient non-zero location')

feature_to_plot = 0
plt.figure()
lw = 2
plt.plot(coef[:, feature_to_plot], color='seagreen', linewidth=lw,
         label='Ground truth')
plt.plot(coef_lasso_[:, feature_to_plot], color='cornflowerblue', linewidth=lw,
         label='Lasso')
plt.plot(coef_multi_task_lasso_[:, feature_to_plot], color='gold', linewidth=lw,
         label='MultiTaskLasso')
plt.legend(loc='upper center')
plt.axis('tight')
plt.ylim([-1.1, 1.1])
plt.show()