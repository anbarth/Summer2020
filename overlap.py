import random
import math
import numpy as np
from logSigmaPlot import makeLogSigmaPlot
from sho import shoEigenket, shoEigenbra

random.seed()

# choose two SHO energy levels
n1 = 0
n2 = 0

# bounds of discretized position space
left = -30
right = 30

# step size
dx = 0.005

# random matrices to avg.
N = 1

#########################################################

# dimension of discretized position space
D = int((right-left)/dx)

# construct the phi and psi matrices
psi = shoEigenket(n1,dx,left,right)
phi = shoEigenbra(n2,dx,left,right)

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