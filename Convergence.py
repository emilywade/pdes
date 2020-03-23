
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt

n_max = 500 # value of N to test to
a = 0 # left interval endpoint
b = 10 # right interval endpoint

Err = [] # to store the error

for n in range(10,n_max):

    h = (b-a)/n

    np.linspace(0,1,10)

    x = np.linspace(a,b,n+1)
    print(x)

    f = np.cos(x) # the function to differentiate
    f_prime = -np.sin(x) # analytical derivative for comparison

    a0 = -np.diag(np.ones(n+1),0)
    a1 = np.diag(np.ones(n),1)
    A = (a0 + a1)/h
    A = A[:-1,:]

    print(A)

    u_prime = A@f # @ is matrix/vector mult from numpy lib
    print(u_prime)

    err = np.max(np.abs(u_prime - f_prime[:-1]))
    
    Err.append(err)
    


# In[4]:

plt.plot(Err)
plt.xlabel('n')
plt.ylabel('Err')
plt.xscale('log')
plt.show()


# In[ ]:



