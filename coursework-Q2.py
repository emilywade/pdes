
# coding: utf-8

# In[8]:

# Solving the BVP u''+ 4u = cos(x) subject to the conditions u(0) = 0, and
# u(1) = 1. Here we use a centered difference approximation to find u

# Imports ---------------------------------------------------------------------

import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------

n = 20 # Number of intervals to split [0,1] into
h = 1/n # Spacing between the nodes
make_plot = 'no' # Switch command to print the plot
plot_name = 'Python1D' # Name to assign the plot

x = np.linspace(0,1,n+1) # Forming the discrete domain

f = h**2*cos(x) # The RHS of the differential equation

# The exact solution of the problem
exact = (sin(2*x)*(3 + cos(2) - cos(1)))/3*sin(2) - (cos(2*x) - cos(x))/3

# Form the finite difference matrix
a = (2*(2*(h**2)-1))*np.diag(np.ones(n+1),0) # The main diagonal (u_i)
b = (1)*np.diag(np.ones(n),1) # The superdiagonal (u_{i+1})
c = (1)*np.diag(np.ones(n),-1) # The subdiagonal (u_{i-1})
A = (a+b+c) # The differentiation matrix

# Modify the problem to include the boundary conditions
A[0,:] = 0; A[0,0] = 1; f[0] = 0 # BC at x=0
A[-1,:] = 0; A[-1,-1] = 1; f[-1] = 1 # BC at x=1

# Solve the linear system
u = np.linalg.solve(A,f)

# Find the error between the exact and approximate solutions
err = max(abs(u - exact))
print('The error in the solution is : ' + str(err))

# Plot the solution
plt.plot(x,exact,'k-')
plt.plot(x,u,'go')
plt.xlabel('x')
plt.ylabel('u')
plt.title('n = ' + str(n) + ', Error = ' + str(round(err,4)))
plt.legend(['Exact Solution','FD Approximation'])
plt.show()

if make_plot == 'yes':
   plt.savefig(plot_name, dpi = 200) 
   



