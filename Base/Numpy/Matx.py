# -*- coding: utf-8 -*-
import numpy as np

# Defining the matrices
A = np.matrix([[3, 6, -5],
               [1, -3, 2],
               [5, -1, 4]])
B = np.matrix([[12],
               [-2],
               [10]])

# Solving for the variables, where we invert A
X = A ** (-1) * B  # A 的逆 乘B  3*3 * 3*1   = 3*1
print(X)

a = np.array([[3, 6, -5],
              [1, -3, 2],
              [5, -1, 4]])
b = np.array([12, -2, 10])
x = np.linalg.inv(a).dot(b)  ## 变成了一行数据
print(x)