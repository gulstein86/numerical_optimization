# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:38:41 2019

@author: vadmin
"""

from scipy.optimize import linprog
import numpy as np

#storing objective function in z variable
#5x1 + 4x2
z=np.array([5,4])

#storing constraints in C variable
C = np.array([
        [1, 1],   #Constraint 1
        [10,6]   #Constraint 2
]) 

#storing upper bound for Constraint 1 and 2 in b variable
b= np.array([5,45])

#specify the soft boundaries for each parameter
x1 = (0, None)
x2 = (0, None)

#Calling Linprog to solve our optimization problem
sol = linprog(-z, A_ub = C, b_ub=b, bounds=(x1, x2), method='simplex')

#printing the result
print(f"x1 = {sol.x[0]}, x2 = {sol.x[1]}, z = {sol.fun*-1}")

# linear programming exercise
#ABC Furniture Shop wants to maximize their profit based on
#THREE (3) products manufactured in their factory, including
#table, chair and book shelve, for the coming week. Table 1
#shows the parts details for these three products and their
#stock availability in the storage, respectively. Each
#completed and manufactured table, chair and book shelve
#will bring profit of RM108, RM60 and RM75 to ABC furniture
#shop.
#Form the objective function and relevant constraints for this problem. 
#z1 = table
#z2 = chair
#z3 = book shelve
#max z = 108x1 + 60x2 + 75x3
#x1 + 0.3x2 + 3x3 = 1000
#4x1 + 4x2 = 3500
#4x3 = 800
#8x1 + 4x2 + 12x3 = 7000


from scipy.optimize import linprog
import numpy as np

#storing objective function in z variable
#5x1 + 4x2
z=np.array([108,60,75])

#storing constraints in C variable
C = np.array([
        [1, 0.3, 3],   #Constraint 1
        [4, 4, 0],   #Constraint 2
        [0, 0, 4],   #Constraint 3
        [8, 4, 12],   #Constraint 4
]) 

#storing upper bound for Constraint 1 and 2 in b variable
b= np.array([1000,3500,800,7000])

#specify the soft boundaries for each parameter
x1 = (0, None)
x2 = (0, None)
x3 = (0, None)

#Calling Linprog to solve our optimization problem
sol = linprog(-z, A_ub = C, b_ub=b, bounds=(x1, x2, x3), method='simplex')

#printing the result
print(f"x1 = {sol.x[0]}, x2 = {sol.x[1]},  x3 = {sol.x[2]},z = {sol.fun*-1}")