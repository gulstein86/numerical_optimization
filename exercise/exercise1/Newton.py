# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 19:21:07 2019

@author: vadmin
"""

import numpy as np
import matplotlib.pyplot as plt
#from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
import time

def func(x):
    return 100*np.square(np.square(x[0])-x[1])+np.square(x[0]-1)

def dfunc(x):
    df1 = 400*x[0]*(np.square(x[0])-x[1])+2*(x[0]-1)
    df2 = -200*(np.square(x[0])-x[1])
    return np.array([df1,df2])

def invhess(x):
    df11 = 1200*np.square(x[0])-400*x[1]+2
    df12 = -400*x[0]
    df21 = -400*x[0]
    df22 = 200
    hess = np.array([[df11,df12],[df21,df22]])
    return np.linalg.inv(hess)

def grad(x, max_int):
    start = time.clock()
    miter = 1
    step = .5
    vals = []
    objectfs = []
    while miter <= max_int:
        vals.append(x)
        objectfs.append(func(x))
        print(x,func(x),miter)
#        temp = x-step*dfunc(x)
        temp = x-step*(invhess(x).dot(dfunc(x)))
        if np.abs(func(temp)-func(x))>0.01: #stop here if the comparison is too small this value is up to us
            x = temp
        else:
            break
        miter +=1
    end = time.clock()
    print ("%.2gs" % (end-start))
    return vals, objectfs, miter



#start = [5,5]
start = [15,15]
val, objectf, iters = grad(start,100)

x = np.array([i[0] for i in val])
y = np.array([i[1] for i in val])
z = np.array(objectf)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x,y,z, label='newton method')
ax.legend()
plt.savefig('newton.jpg')
