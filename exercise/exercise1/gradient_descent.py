# -*- coding: utf-8 -*-
"""
Spyder Editor

author: aswadi
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def func(x):
    return 100*np.square(np.square(x[0])-x[1])+np.square(x[0]-1) #calculate the z

def dfunc(x):
    df1 = 400*x[0]*(np.square(x[0])-x[1])+2*(x[0]-1)
    df2 = -200*(np.square(x[0])-x[1])
    return np.array([df1,df2])

def grad(x, max_int):
    start = time.clock()
    miter = 1
    step = .0001/miter
    vals = []
    objectfs = []
    while miter <= max_int:
        vals.append(x)
        objectfs.append(func(x))
        print(x,func(x),miter)
        temp = x-step*dfunc(x)  # 
        if np.abs(func(temp)-func(x))>0.01: #stop here if the comparison is too small this value is up to us
            x = temp
        else:
            break        
        miter +=1
    end = time.clock()
    print ("%.2gs" % (end-start))
    return vals, objectfs, miter


start = [5,5]
# start = [15,15]
val, objectf, iters = grad(start,50)
start_time=time.time()

x = np.array([i[0] for i in val])
y = np.array([i[1] for i in val])
z = np.array(objectf)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x,y,z, label='gradient descent method')
ax.legend()
plt.savefig('GradientDescent.jpg')
