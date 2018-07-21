# -*- coding: utf-8 -*-

import numpy as np

# Creating an array for demonstration
array = np.arange(10)
function_mean = np.mean(array)
method_mean = array.mean()
print(function_mean)
print(method_mean)

coth = lambda x: 1 / np.tanh(x)   # Hyperbolic cotangent
sech = lambda x: 2 / (np.exp(x) - np.exp(-x))  # Hyperbolic secant
arccsch = lambda x: np.log(1 / x + np.sqrt(1 + x ** 2) / np.abs(x))  # An inverse hyperbolic cosecant
print(coth(20))

array = np.random.rand(100)
array[5] = np.nan
print(np.max(array))   # Returns inccorrect result  # 只要含有一个nan 整个就是 nan
print(np.nanmax(array))  # Returns correct result

array = np.random.rand(10000).reshape((100, 100))


rindex = np.random.randint(2, size=(100, 100))  # Throw NaNs in random places in an array
array[rindex] = np.nan
not_nan = ~np.isnan(array)   # Creating index of non-NaN values  是一个 boolean 矩阵
print(not_nan)
print(np.std(array[not_nan]))  # Now using functions that cannot handle NaNs, which # returns the correct standard deviation value.

