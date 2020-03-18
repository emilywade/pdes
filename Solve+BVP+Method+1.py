
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt

from numpy import cos, sin, pi 

n = 100 # number of discrete intervals
h = 1/n # spacing between the nodes

x = np.linspace(0,1,n+1)

# form the finite difference matrix - method 1
a = -2*np.diag(np.ones(n-1),0) # np.ones(n-1) gives n-1 vector of 1s, then np.diag puts it across diagonal, 0 elsewhere
b = np.diag(np.ones(n-2),1) # superdiagonal
c = np.diag(np.ones(n-2),-1) # subdiagonal

A = a+b+c

# determine RHS of problem
f = h**2*pi**2*sin(pi*x[1:-1])
f[0] = f[0]-0
f[-1] = f[-1] - 1

# answer in the interior nodes
u = np.linalg.solve(A,f)
u = np.concatenate(([0],u,[1]))

u_exact = x-sin(pi*x)

# plot the solution 
plt.plot(x,u_exact,'k-')
plt.plot(x,u,'b--')
plt.show()

# check error
err = max(abs(u-u_exact))
print('The error is' + str(err))


# In[ ]:



