
# coding: utf-8

# In[2]:

import numpy as np
import matplotlib.pyplot as plt

n = 200 # number of discrete intervals
a = 0 # left interval endpoint
b = 10 # right interval endpoint

h = (b-a)/n

x = np.linspace(a,b,n+1)

f = np.cos(x) # the function to differentiate
f_prime = -np.sin(x) # analytical derivative for comparison

a0 = -np.diag(np.ones(n+1),0)
a1 = np.diag(np.ones(n),1)
A = (a0 + a1)/h
A = A[:-1,:]

u_prime = A@f # @ is matrix/vector mult from numpy lib

# we have found a derivative and now we want to check it is right

plt.plot(x[:-1],u_prime,'k-')
plt.plot(x,f_prime,'b--')
plt.show()

# error at end can be fixed by truncating the matrix 
# will always be slight error since this is only order h accurate


# In[ ]:



