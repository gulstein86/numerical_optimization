import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# the objective function
def func(x):
    return 100*np.square(np.square(x[0])-x[1])+np.square(x[0]-1)

# first order derivatives of the function
def dfunc(x):
    df1 = 400*x[0]*(np.square(x[0])-x[1])+2*(x[0]-1)
    df2 = -200*(np.square(x[0])-x[1])
    return np.array([df1, df2])

# the armijo algorithm
def armijo(valf, grad, niters):
    #beta = random.random()
    #sigma = random.uniform(0, .5)
    beta = 0.25
    sigma = 0.25
    (miter, iter_conv) = (0, 0)
    conval = [0,0]
    val = []
    objectf = []
    val.append(valf)
    objectf.append(func(valf))
    while miter <= niters:
        leftf = func(valf+np.power(beta, miter)*grad)
        rightf = func(valf) + sigma*np.power(beta, miter)*dfunc(valf).dot(grad)
        if leftf-rightf <= 0:
            iter_conv = miter
            conval = valf+np.power(beta, iter_conv)*grad
            break
        miter += 1
        val.append(conval)
        objectf.append(func(conval))
    return conval, func(conval), iter_conv, val, objectf

# initialization
start = np.array([-.3, .1])
direction = np.array([1, -2])
maximum_iterations = 30

converge_value, minimal, no_iter, val, objf = armijo(start, direction, maximum_iterations)
print("The value, minimal and number of iterations are " + str(converge_value) + \
    ", " + str(minimal) + ", " + str(no_iter))
x = np.array([i[0] for i in val])
y = np.array([i[1] for i in val])
z = np.array(objf)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x, y, z, label='Armijo Rule')
ax.legend()
plt.savefig('armijo.jpg')