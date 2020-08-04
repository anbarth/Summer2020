import math
import numpy as np
import matplotlib.pyplot as plt
import sho
import importlib

importlib.reload(sho)

# choose two SHO energy levels
n1 = 10
#n2 = 0

# bounds of discretized position space
left = -20
right = 20
plotLeft = -10
plotRight = 10

# step size
dx = 0.1
eigendx = 0.01

#########################################################

# dimension of discretized position space
D = int((right-left)/dx)

# construct the phi and psi matrices
depth = 0
width = 1
center = 0
(E,psis) = sho.defectEigenstates(depth,width,center,left,right,dx,n1,n1,eigendx)
#(E,psis) = sho.oldDefectEigenstates(depth,width,center,left,right,dx,n1,n1)
psi = psis[0]

domain = np.linspace(left,right,D,endpoint=False)


#overlap = 0
#for i in range(D):
#    overlap += psi[i] * phi[i]
#print(overlap)

plt.clf()
plt.xlim(plotLeft,plotRight)
#plt.title('n1='+str(n1)+', n2='+str(n2)+'. dx='+str(dx)+' over ['+str(left)+','+str(right)+']')
plt.title('n='+str(n1)+'. dx='+str(dx)+' over ['+str(left)+','+str(right)+']')
plt.plot(domain,psi)

#plt.hist(domain,bins=len(domain),weights=psi)
plt.show()