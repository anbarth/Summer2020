import random
import numpy as np
import csv
import scipy.special
import math
import time
import statistics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

tic = time.perf_counter()
random.seed()

# choose two SHO energy levels
n1 = 1
n2 = 2

# bounds of discretized position space
left = -20
right = 20

# dimension of discretized position space
dx = 5
D = int((right-left)/dx)

# options for number of random matrices to avg.
Nlist = [10,50,250,1250,5000] 

# number of trials to take for each value of N
trials = 10

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
    # a row of results to record in my table, in format:
    # 0 |  1   |        ...             | -3  |   -2    |   -1
    # N | ln N | ... overlap trials ... | avg | std dev | ln std dev
    results = [N]
    results.append(np.log(N))

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
    
    # add summary statistics to this row of the table
    # average overlap:
    results.append(statistics.mean(results[2:])) 
    # std dev of overlaps:
    results.append(statistics.stdev(results[2:-1]))
    # ln std dev of overaps:
    results.append(np.log(results[-1]))
    resultsTable.append(results)

# construct header rows
infoRow = ['n1='+str(n1)+'; n2='+str(n2)+'. dx='+str(dx)+' over ['+str(left)+','+str(right)+']']
headerRow = ['N','ln N']
for i in range(trials):
    headerRow.append(' ')
headerRow.extend(['avg','sigma','ln sigma'])

with open('overlaps.csv','w',newline='') as csvFile:
    writer = csv.writer(csvFile, delimiter=',')
    writer.writerow(infoRow)
    writer.writerow(headerRow)
    for row in resultsTable:
        writer.writerow(row)

lnN = [resultsTable[i][1] for i in range(len(resultsTable))]
lnSigma = [resultsTable[i][-1] for i in range(len(resultsTable))]
plt.plot(lnN,lnSigma,'.',color='black')

lnN_arr = np.array(lnN).reshape((-1,1))
lnSigma_arr = np.array(lnSigma)
model = LinearRegression()
model.fit(lnN_arr,lnSigma_arr)

slope = model.coef_
intercept = model.intercept_
r_sq = model.score(lnN_arr,lnSigma_arr)
print("slope: "+str(slope))
print("intercept: "+str(intercept))
print("R^2: "+str(r_sq))

lnSig_model = slope*lnN_arr + intercept
plt.plot(lnN,lnSig_model)

toc = time.perf_counter()
print("runtime "+str(toc-tic))
plt.show()

