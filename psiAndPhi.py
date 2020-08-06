import math
import numpy as np
import matplotlib.pyplot as plt
import sho
import importlib

importlib.reload(sho)

# choose two SHO energy levels
n1 = 2
#n2 = 0

# bounds of discretized position space
left = -100
right = 100
plotLeft = -50
plotRight = 50

# step size
dx = 0.05
eigendx = 0.05


#########################################################

# dimension of discretized position space
D = int((right-left)/dx)


# construct the phi and psi matrices
depth = 10
width = 1
center = 0
#(E,psis) = sho.defectEigenstates(depth,width,center,left,right,dx,n1,n1,eigendx)
(E,psis) = sho.squareWellEigenstates(depth,left,right,dx,n1,n1,eigendx)

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
plt.plot(domain,psi,color='blue')


#plt.hist(domain,bins=len(domain),weights=psi)
plt.show()