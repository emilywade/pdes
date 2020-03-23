
# coding: utf-8

# In[3]:

import numpy as np
import matplotlib.pyplot as plt

n_max = 500 # final value of n to increment to
a = 0 # left interval endpoint
b = 10 # right interval endpoint

# initialise empty lists to store error
Err_forward = []
Err_backward = []
Err_centred = []

for n in range(10,n_max+1):
    h = (b-a)/n # distance between nodes
    x = np.linspace(a,b,n+1) # generate n+1 discrete intervals
    f = np.cos(x) # function to differentiate
    f_prime = -np.sin(x) # analytical derivative for comparison
    
    # find the forward difference approximation
    a0 = -np.diag(np.ones(n+1),0) # main diagonal
    a1 = np.diag(np.ones(n),1) # super diagonal
    A = (a0 + a1) / h # form differentiation matrix
    A = A[:-1,:] # removing last row of matrix 
    
    u_prime = A@f # find approximate derivative
    f_prime = -np.sin(x[:-1]) # find analytical derivative
    
    # find the error in the solution
    Err_forward.append(max(abs(u_prime-f_prime)))
    # ------------------------------------------------------------------
    
    # find the backward difference approximation
    a0 = np.diag(np.ones(n+1),0) # main diagonal
    a1 = -np.diag(np.ones(n),-1) # sub diagonal
    A = (a0 + a1) / h # form differentiation matrix
    A = A[1:,:] # removing first row of matrix 
    
    u_prime = A@f # find approximate derivative
    f_prime = -np.sin(x[1:]) # find analytical derivative
    
    # find the error in the solution
    Err_backward.append(max(abs(u_prime-f_prime)))
    # ------------------------------------------------------------------
 
    # find the centred difference approximation
    a0 = np.diag(np.ones(n),1) # super diagonal
    a1 = -np.diag(np.ones(n),-1) # sub diagonal
    A = (a0 + a1) / (2*h) # form differentiation matrix
    A = A[1:-1,:] # removing first and last rows of matrix 
    
    u_prime = A@f # find approximate derivative
    f_prime = -np.sin(x[1:-1]) # find analytical derivative
    
    # find the error in the solution
    Err_centred.append(max(abs(u_prime-f_prime)))
    # ------------------------------------------------------------------
    
plt.plot(Err_forward,'k-',linewidth = 1) # plotting the foward difference error
plt.plot(Err_backward,'b--') # plotting the backward difference error
plt.plot(Err_centred,'r:') # plotting the centred difference error
plt.xlabel('n')
plt.ylabel('Error')
plt.legend(('Forward Difference','Backward Difference','Centred Difference'))
plt.title('Error Convergence as n Increases')
plt.xscale('log')
plt.show()


# In[ ]:

#centred difference approximation gives a much smaller error which is desirable

