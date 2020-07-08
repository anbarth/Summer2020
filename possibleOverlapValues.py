import math
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed()

# choose two SHO energy levels
n1 = 0
n2 = 0

# bounds of discretized position space
left = -1
right = 1

# step size
dx = 0.025

# num values to put in histogram
N = 10000

#########################################################

# dimension of discretized position space
D = int((right-left)/dx)

# construct the phi and psi matrices
psi = []
phi = []
domain = []

# make hermite polynomial objects
n1_arr = [0]*(n1+1)
n1_arr[-1] = 1
n2_arr = [0]*(n2+1)
n2_arr[-1] = 1
herm1 = np.polynomial.hermite.Hermite(n1_arr,[left,right])
herm2 = np.polynomial.hermite.Hermite(n2_arr,[left,right])
herm1_arr = herm1.linspace(n=D)[1]
herm2_arr = herm2.linspace(n=D)[1]

# psi and phi's norm-squareds, so i can normalize later
norm1 = 0
norm2 = 0

x = left
for i in range(D):
    domain.append(x)
    psi.append(math.exp(-1*x*x)*herm1_arr[i])
    phi.append(math.exp(-1*x*x)*herm2_arr[i])
    norm1 += psi[i]*psi[i]
    norm2 += phi[i]*phi[i]
    x += dx
# normalize
psiN = [x*(1/math.sqrt(norm1)) for x in psi]
phiN = [x*(1/math.sqrt(norm2)) for x in phi]

psi = psiN
phi = phiN

#### overwrite
#D = 3
#psi = [1]*D
#phi = [1]*D

####

overlaps = []
for i in range(N):
    zeta = [[random.choice([-1,1])] for i in range(D)]
    phizeta = np.matmul(phi, zeta) # <phi|z>
    zetapsi = np.matmul(np.transpose(zeta), psi) # <z|psi>
    prod = phizeta * zetapsi
    overlaps.append(prod[0])

plt.title('n1='+str(n1)+'; n2='+str(n2)+'. dx='+str(dx)+' over ['+str(left)+','+str(right)+']')
plt.hist(overlaps)
plt.show()