import math
import numpy as np
import matplotlib.pyplot as plt

# choose two SHO energy levels
n1 = 10
n2 = 0

# bounds of discretized position space
left = -3
right = 3

# step size
dx = 0.025

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
norm1 = 10
norm2 = 10

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

plt.hist(domain,weights=psi)
plt.show()