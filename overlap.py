import random
import numpy as np
import csv
import scipy.special
import math


random.seed()

# choose two SHO energy levels
n1 = 1
n2 = 2

# bounds of discretized position space
left = -10
right = 10

# dimension of discretized position space
D = 80

# options for number of random matrices to avg.
#Nlist = [10,100,500,1000,2500,5000] 
Nlist = [100]

# construct the phi and psi matrices
psi = np.zeros((D,1))
phi = np.zeros((1,D))
# psi and phi's norm-squareds, so i can normalize later
norm1 = 0
norm2 = 0

dx = (right-left)/D
x = left
for i in range(D):
    psi[i][0] = math.exp(-1*x*x)*scipy.special.eval_hermite(n1, x)
    phi[0][i] = math.exp(-1*x*x)*scipy.special.eval_hermite(n2, x)
    norm1 += psi[i][0]*psi[i][0]
    norm2 += phi[0][i]*phi[0][i]
    x += dx
# normalize
psi = psi*(1/math.sqrt(norm1))
phi = phi*(1/math.sqrt(norm2))

resultsTable = []
for N in Nlist:
    # a row of results to record in my table
    # first entry in the row is N
    results = [N]

    runningTot = 0
    for i in range(N):
        zeta = [[random.choice([-1,1])] for i in range(D)]
        phizeta = np.matmul(phi, zeta) # <phi|z>
        zetapsi = np.matmul(np.transpose(zeta), psi) # <z|psi>
        prod = phizeta * zetapsi
        runningTot = runningTot + prod

    runningTot = runningTot*(1/N) 
    print(runningTot[0][0])
    
    
    resultsTable.append(results)