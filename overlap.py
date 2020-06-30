import random
import numpy as np
import csv
import scipy.special
import math
import time

tic = time.perf_counter()
random.seed()

# choose two SHO energy levels
n1 = 1
n2 = 2

# bounds of discretized position space
left = -20
right = 20

# dimension of discretized position space
dx = 0.025
D = int((right-left)/dx)

# options for number of random matrices to avg.
Nlist = [10,50,250,1250,5000] 

# number of trials to take for each value of N
trials = 25

# construct the phi and psi matrices
psi = np.zeros((D,1))
phi = np.zeros((1,D))
# psi and phi's norm-squareds, so i can normalize later
norm1 = 0
norm2 = 0

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

    for j in range(trials):
        runningTot = 0
        for i in range(N):
            zeta = [[random.choice([-1,1])] for i in range(D)]
            phizeta = np.matmul(phi, zeta) # <phi|z>
            zetapsi = np.matmul(np.transpose(zeta), psi) # <z|psi>
            prod = phizeta * zetapsi
            runningTot = runningTot + prod

        runningTot = runningTot*(1/N) 
        results.append(runningTot[0][0])
    
    resultsTable.append(results)

with open('overlaps.csv','w',newline='') as csvFile:
    writer = csv.writer(csvFile, delimiter=',')
    for row in resultsTable:
        writer.writerow(row)

toc = time.perf_counter()
print("runtime "+str(toc-tic))