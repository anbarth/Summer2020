import math
import numpy as np
import matplotlib.pyplot as plt
import sho
import importlib

importlib.reload(sho)

# choose two SHO energy levels
n1 = 13
#n2 = 0

# bounds of discretized position space
left = -20
right = 20
plotLeft = -5
plotRight = 5

# step size
dx = 0.05
eigendx = 0.01
dx2 = 0.01
eigendx2 = 0.01

#########################################################

# dimension of discretized position space
D = int((right-left)/dx)
D2 = int((right-left)/dx2)

# construct the phi and psi matrices
depth = 1000
width = 2
center = 0
(E,psis) = sho.defectEigenstates(depth,width,center,left,right,dx,n1,n1,eigendx)
(E2,psis2) = sho.defectEigenstates(depth,width,center,left,right,dx2,n1,n1,eigendx2)
#(E,psis) = sho.oldDefectEigenstates(depth,width,center,left,right,dx,n1,n1)
psi = psis[0]
psi2 = psis2[0]

domain = np.linspace(left,right,D,endpoint=False)
domain2 = np.linspace(left,right,D2,endpoint=False)

#overlap = 0
#for i in range(D):
#    overlap += psi[i] * phi[i]
#print(overlap)

plt.clf()
plt.xlim(plotLeft,plotRight)
#plt.title('n1='+str(n1)+', n2='+str(n2)+'. dx='+str(dx)+' over ['+str(left)+','+str(right)+']')
plt.title('n='+str(n1)+'. dx='+str(dx)+' over ['+str(left)+','+str(right)+']')
plt.plot(domain,psi,color='blue')
plt.plot(domain2,psi2,color='orange')

#plt.hist(domain,bins=len(domain),weights=psi)
plt.show()