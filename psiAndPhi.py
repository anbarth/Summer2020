import math
import numpy as np
import matplotlib.pyplot as plt
from sho import shoEigenket

# choose two SHO energy levels
n1 = 0
n2 = 0

# bounds of discretized position space
left = -30
right = 30

# step size
dx = 0.05

#########################################################

# dimension of discretized position space
D = int((right-left)/dx)

# construct the phi and psi matrices
psi = shoEigenket(n1,dx,left,right)
phi = shoEigenket(n2,dx,left,right)
domain = np.linspace(left,right,D,endpoint=False)


overlap = 0
for i in range(D):
    overlap += psi[i] * phi[i]
print(overlap)

plt.title('n1='+str(n1)+', n2='+str(n2)+'. dx='+str(dx)+' over ['+str(left)+','+str(right)+']')
plt.plot(domain,psi)
plt.plot(domain,phi)
#plt.hist(domain,bins=len(domain),weights=psi)
plt.show()