import random
import math
import numpy as np
from logSigmaPlot import makeLogSigmaPlot

random.seed()

# choose two SHO energy levels
n1 = 0
n2 = 0

# bounds of discretized position space
left = -12
right = 12

# step size
dx = 2

# random matrices to avg.
N = 1

#########################################################

# dimension of discretized position space
D = int((right-left)/dx)

# construct the phi and psi matrices
psi = np.zeros((D,1))
phi = np.zeros((1,D))

# make hermite polynomial objects
n1_arr = [0]*(n1+1)
n1_arr[-1] = 1
n2_arr = [0]*(n2+1)
n2_arr[-1] = 1
herm1 = np.polynomial.hermite.Hermite(n1_arr,window=[left,right])
herm2 = np.polynomial.hermite.Hermite(n2_arr,window=[left,right])
herm1_arr = herm1.linspace(n=D)[1]
herm2_arr = herm2.linspace(n=D)[1]

# psi and phi's norm-squareds, so i can normalize later
norm1 = 0
norm2 = 0

x = left
for i in range(D):
    psi[i][0] = math.exp(-1*x*x)*herm1_arr[i]
    phi[0][i] = math.exp(-1*x*x)*herm2_arr[i]
    norm1 += psi[i][0]*psi[i][0]
    norm2 += phi[0][i]*phi[0][i]
    x += dx
# normalize
psi = psi*(1/math.sqrt(norm1))
phi = phi*(1/math.sqrt(norm2))

# find <phi|psi> directly, for reference
trueOverlap = np.matmul(phi,psi)
print(trueOverlap[0][0])

overlap = 0
for i in range(N):
    zeta = [[random.choice([-1,1])] for i in range(D)]
    phizeta = np.matmul(phi, zeta) # <phi|z>
    zetapsi = np.matmul(np.transpose(zeta), psi) # <z|psi>
    prod = phizeta * zetapsi
    overlap = overlap + prod

overlap = overlap*(1/N) 
print(overlap[0][0])