
# coding: utf-8

# 1. Why the "+"? 
# 2. Identify ONE technique / method (not in lecture) to perform step length calculation. 
# 3. Identify ONE technique / method (not in lecture) to perform direction searching. 
# 4. Explain the practical steps in Armijo function using words. 
# 5. Explain the practical steps in Wolfe function using words. 
# 6. What is your observation for the results generated using Armijo conditions and wolfe conditions? Discuss.

# In[1]:


import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# the objective function
def func(x):
    return 100*np.square (np.square(x[0])-x[1])+np.square(x[0]-1)
                                                       
# first order derivatives of the function
def dfunc(x):
    df1 = 400*x[0]*(np.square(x[0])-x[1])+2*(x[0]-1)
    df2 = 200*(np.square(x[0])-x[1])
    return np.array([df1, df2])


# In[8]:


# the armijo algorithm
def armijo (valf, grad, niters):
    #beta random.random()
    #sigma random. uniform(0, .5)
    beta= 0.25
    sigma= 0.25
    (miter, iter_conv) =(0, 0)
    conval = [0,0]
    val = []
    objectf = []
    val.append(valf)
    objectf.append (func(valf))
    while miter <= niters:
        leftf = func(valf+np.power(beta, miter)*grad)
        rightf = func(valf) + sigma *np.power(beta, miter)*dfunc(valf).dot(grad)
        if leftf-rightf <=0:
            iter_conv = miter
            conval = valf+np.power(beta, iter_conv)*grad
            break
        miter +=1
        val.append (conval)
        objectf.append (func (conval))
    return conval, func (conval), iter_conv, val, objectf


# In[11]:


# initialization
start = np.array([-.3, .1],)
direction = np.array ([1, -2])
maximum_iterations = 30

converge_value, minimal, no_iter, val, objf = armijo(start, direction, maximum_iterations)
print("The value, minimal and number of iterations are " + str(converge_value) + ", " + str(minimal)+ ", " + str(no_iter))
x = np.array([i[0] for i in val])
y = np.array([i[1] for i in val])
z = np.array (objf)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x, y, z, label='Armijo Rule')
ax.legend()
plt.savefig('armijo.jpg')


# In[33]:


def wolfe(valf, direction, max_iter):
    (alpha, beta, step, c1, c2) = (0, 1000, 5.0, 0.15, 0.3)
    i = 0
    stop_iter = 0
    stop_val = valf
    minima = 0
    val = []
    objectf = []
    val.append (valf)
    objectf.append (func(valf))
    while i <= max_iter:
        # first confition
        leftf = func(valf + step*direction)
        rightf = func(valf) + step*c1*dfunc(valf).dot(direction)
        if leftf > rightf:
            beta = step
            step = .5*(alpha + beta)
            val.append (valf+step*direction)
            objectf.append(leftf)
        elif dfunc (valf + step*direction).dot(direction) < c2*dfunc(valf).dot(direction):
            alpha = step
            if beta > 100:
                step = 2*alpha
            else:
                step = .5*(alpha + beta)
            val.append (valf+step*direction)
            objectf.append(leftf)
        else:
            val.append (valf+step*direction)
            objectf.append(leftf)
            break
        i+= 1
        stop_val = valf + step*direction
        stop_iter = i
        minima = func(stop_val)
    print(val, objectf)
    return stop_val, minima, stop_iter, step, val, objectf


# In[34]:


start = np.array([.6, .5])
dirn = np.array([-.3, -.4])

converge_value, minimal, no_iter, size, val, objectf = wolfe(start, dirn, 30)
print("The value, minimal and iterations needed are " + str(converge_value) +", " +str(minimal) + ", " + str(no_iter) + ", "+ str(size))
x = np.array ([i[0] for i in val])
y = np.array ([i[1] for i in val])
z = np.array (objectf)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x, y, z, label='Wolfe Rule')
ax.legend()
plt.savefig('wolfe.jpg')

