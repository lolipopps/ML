# -*- coding: utf-8 -*-
import numpy as np
import timeit
alist = [1, 2, 3]
arr = np.array(alist)
arr = np.zeros(5) # Creating an array of zeros with five elements
arr = np.arange(100)   # What if we want to create an array going from 0 to 100?
arr = np.arange(10, 100)  # Or 10 to 100?
arr = np.linspace(0, 1, 100)   # If you want 100 steps from 0 to 1...
arr = np.logspace(0, 1, 100, base=10.0)  # Or if you want to generate an array from 1 to 10 # in log10 space in 100 steps...
image = np.zeros((5, 5))  # Creating a 5x5 array of zeros (an image)
cube = np.zeros((5, 5, 5)).astype(int) + 1  # Creating a 5x5x5 cube of 1's # The astype() method sets the array with integer elements.
cube = np.ones((5, 5, 5)).astype(np.float16)  # Or even simpler with 16-bit floating-point precision...
arr = np.zeros(2, dtype=int) # Data typing # Array of zero integers
arr = np.zeros(2, dtype=np.float32)  # Array of zero floats
arr1d = np.arange(1000)   # Reshaping # Creating an array with elements from 0 to 999
arr3d = arr1d.reshape((10, 10, 10))  # Now reshaping the array to a 10x10x10 3D array
arr3d = np.reshape(arr1d, (10, 10, 10))  # The reshape command can alternatively be called this way
arr4d = np.zeros((10, 10, 10, 10))  # Inversely, we can flatten arrays
arr1d = arr4d.ravel()   ## 变成一行
print(arr1d.shape)
recarr = np.zeros((2,), dtype=('i4,f4,a10'))  ## 指定输入类型  ## [(0, 0., b'') (0, 0., b'')]
toadd = [(1, 2., "Hello"), (2, 3., "World")]
recarr[:] = toadd   # [(1, 2., b'Hello') (2, 3., b'World')]
alist = [[1, 2], [3, 4]]
print(alist[0][1])  # 2

arr = np.array(alist)
print(arr[0, 1]) # 2 返回第一行第二列

print(arr[:, 1]) # [2,4]  返回d第二列  :,代表所有行
print(arr[1, :])  #返回第二行
index = np.where(arr > 1)
print(index)
# Creating an array
arr = np.arange(5)

# Creating the index array
index = np.where(arr > 2)
print(index)
arr = np.arange(1e7)  # Create an array with 10^7 elements.
larr = arr.tolist()  # Converting ndarray to list
# Lists cannot by default broadcast, so a function is coded to emulatewhat an ndarray can do.
def list_times(alist, scalar):
    for i, val in enumerate(alist):
        alist[i] = val * scalar
    return alist
N = 10  # Number of tries

# We are not using IPython's magic timeit command here. This enables you to
# run the script in as a script.
# NumPy array broadcasting
time1 = timeit.timeit('arr * 1.1', 'from __main__ import arr', number=N) / N
print(time1)
# List and custom function for broadcasting
time2 = timeit.timeit('list_times(larr, 1.1)',
	'from __main__ import list_times, larr', number=N) / N
print(time2)