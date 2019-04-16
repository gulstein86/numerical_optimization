# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:03:59 2019

@author: Ong Sim Ying
"""

import numpy as np
import matplotlib.pyplot as plt
from linalg_utils import *
import trust_region as tr
import step_finders as sf

f1 = lambda x: 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
g1 = lambda x: np.array([-400*(x[1] - x[0]**2)*x[0] - 2*(1-x[0]), 200*(x[1] - x[0]**2)])
h1 = lambda x: np.array([
    [-400*(x[1] - 3*x[0]**2) + 2, -400*x[0]],
    [-400*x[0], 200]]
)
x1s = [np.array([2, 1]), np.array([0, 1]), np.array([-1, 1])]

a2 = lambda x: np.exp(-1. / (100*(x[0]-1))**2)
f2 = lambda x: x[0]**2 + a2(x) - 1
g2 = lambda x: np.array([2*x[0] + a2(x)*200*(1./(100*(x[0]-1))**3)])
h2 = lambda x: np.array([2 + a2(x)*((200*(1./(100*(x[0]-1))**3))**2 + 600*(1./(100*(x[0]-1))**4))])
x2s = [np.array([-1]), np.array([-2])]

x, fx, iterations = tr.trust_region(f2, g2, h2, x2s[0], .1, 1, .15, sf.dogleg_step_finder)
print('Result in %d iterations:' % iterations)
print('%s -> %f' % (x, fx))