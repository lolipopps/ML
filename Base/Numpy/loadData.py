# -*- coding: utf-8 -*-
import numpy as np
arr = np.loadtxt('./data/somefile.txt')  # Loading and existing file
np.savetxt('./data/somenewfile.txt', arr)  # Saving a new file
f = open('./data/existingfile.txt', 'a')  # Opening an existing file with the append option
data2append = np.random.rand(100)  # Creating some random data to append to the existing file
np.savetxt(f, data2append)   # With np.savetxt we replace the file name with the file handle.
f.close()
table = np.loadtxt('./data/example.txt',
		dtype={'names': ('ID', 'Result', 'Type'),
'formats': ('S4', 'f4', 'i2')})
print(table.real)

data = np.empty((1000, 1000))
np.save('./data/test.npy', data)
np.savez('./data/test.npz', data)
newdata = np.load('./data/test.npz')
