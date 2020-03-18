
# coding: utf-8

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, sin, pi 

n = 20 # number of discrete intervals
h = 1/n # spacing between the nodes

x = np.linspace(0,1,n+1)

#form RHS and exact soln
f = h**2*pi**2*sin(pi*x) # RHS of differential eqn
u_exact = x-sin(pi*x) # exact soln

# form the finite difference matrix - method 2
a = -2*np.diag(np.ones(n+1),0) # np.ones(n-1) gives n-1 vector of 1s, then np.diag puts it across diagonal, 0 elsewhere
b = np.diag(np.ones(n),1) # superdiagonal
c = np.diag(np.ones(n),-1) # subdiagonal

A = a+b+c

# modify the problem to include BCs
A[0,:] = 0; A[0,0] = 1; f[0] = 0 # BC at x=0
A[-1,:] = 0; A[-1,-1] = 1; f[-1] = 1 # BC at x=1

# solve the linear system
u = np.linalg.solve(A,f)

# check error
err = max(abs(u-u_exact))
print('The error is' + str(err))

# plot the solution 
plt.plot(x,u_exact,'k-')
plt.plot(x,u,'b--')
plt.xlabel('x')
plt.ylabel('u')
plt.show()


# In[ ]:



