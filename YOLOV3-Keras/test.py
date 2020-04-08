"""
    @Author: Tan.wt
    @Time: 2018/05/25
    @File: test.py
    @License: Apache License
"""
import numpy as np

l = [2, 1,0,4,5,7,1,5,6]
a = np.asarray(l).reshape([3,3])
print(a)

print(np.argmax(a, axis=-1))
print(np.argmax(a, axis=0))

print(np.argmax(a, axis=-2))