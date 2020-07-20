import time
import random
import numpy as np
from sho import shoEigenbra
from myStats import mean,stdev
from linReg import regress
import matplotlib.pyplot as plt
import csv
import multiprocessing as mp



tic = time.time()
random.seed()

### SET UP
nMax = 10 # inclusive
left = -20
right = 20
dx = 0.05
Nmax = 5000
#sampleSize = 50
trials = 100

# dimension of discretized position space
D = int((right-left)/dx)

# get all eigenfunctions
eigens = np.zeros((nMax+1,D))
for n in range(nMax+1):
    eigens[n] = shoEigenbra(n,dx,left,right)


### THE MEAT
overlaps = np.zeros((nMax+1,nMax+1,Nmax,trials))
for i in range(trials):
    print("trial "+str(i))
    psizeta = np.zeros((nMax+1,Nmax))
    for N in range(1,Nmax+1):
        zeta = [random.choice([-1,1]) for x in range(D)] # <z|
        #col = np.zeros((nMax+1))
        for n in range(nMax+1):
            # TODO some complex conjugate nonesense might be needed here
            psizeta[n][N-1]=(np.dot(eigens[n], zeta)) # <psi_n|z>
        for n1 in range(nMax+1):
            for n2 in range(n1,nMax+1):
                overlaps[n1][n2][N-1][i] = np.vdot(psizeta[n1], psizeta[n2])*(1.0/N)

intercepts = np.zeros((nMax+1,nMax+1))
intercept_errs = np.zeros((nMax+1,nMax+1))
lnN = [np.log(N) for N in range(1,Nmax+1)]
lnSigma = np.zeros((nMax+1,nMax+1,Nmax))
for n1 in range(nMax+1):
    for n2 in range(n1,nMax+1):
        for N in range(1,Nmax+1):
            lnSigma[n1][n2][N-1] = np.log(stdev(overlaps[n1][n2][N-1]))
        
        (slope, intercept, r_sq, slope_err, intercept_err) = regress(lnN, lnSigma[n1][n2])
        intercepts[n1][n2] = intercept
        intercept_errs[n1][n2] = intercept_err


with open('heatmap.csv','w') as csvFile:
    writer = csv.writer(csvFile,delimiter=',')

    writer.writerow(['dx='+str(dx)+' over ['+str(left)+','+str(right)+']'])
    writer.writerow(['max N: '+str(Nmax)+', trials: '+str(trials)])

    writer.writerow(['intercepts'])
    for i in range(len(intercepts)):
        writer.writerow(intercepts[i])

    writer.writerow(['intercept errors'])
    for i in range(len(intercept_errs)):
        writer.writerow(intercept_errs[i])

toc = time.time()
print('runtime (s): '+str(toc-tic))
