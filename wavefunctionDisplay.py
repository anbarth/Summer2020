import math
import numpy as np
import matplotlib.pyplot as plt
import sho
import importlib
importlib.reload(sho)

# this script displays one or two energy eigenfunctions

############ INPUTS ####################

# choose two energy levels
n1 = 0
n2 = 2

# bounds of discretized position space
left = -20
right = 20

# potential parameters
depth = 500
width = 1
center = 0

# bounds of plot
plotLeft = -1.5
plotRight = 1.5

# mesh size...
dx = 0.01 # ...for wavefxns
dx_solve = dx # ...for solving schrodinger eqn


#########################################################

# dimension of discretized position space
D = int((right-left)/dx)

# get those eigenfunctions
(E,psis) = sho.defectEigenstates(depth,width,center,left,right,min(n1,n2),max(n1,n2),dx,dx_solve)
psi = psis[0]
phi = psis[-1]

a = np.dot(psi,psi)
print(a)

# plot them!
domain = np.linspace(left,right,D,endpoint=False)

plt.clf()
plt.xlim(plotLeft,plotRight)
#plt.title('n1='+str(n1)+', n2='+str(n2)+'. dx='+str(dx)+' over ['+str(left)+','+str(right)+']')
plt.title('n='+str(n1)+'. dx='+str(dx)+' over ['+str(left)+','+str(right)+']')

# dotted line, with markers
#plt.plot(domain,psi,color='gray',linestyle='dotted',marker='o',markeredgecolor='black',markerfacecolor='black',markevery=200)

# solid line
plt.plot(domain,psi,color='black')
#plt.plot(domain,phi,color='blue')

# make the plot out of boxes with width dx
#plt.hist(domain,bins=len(domain),weights=psi)

plt.show()