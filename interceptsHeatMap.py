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
trials = 50
#numTrialGroups = 10
#trialGroupSize = 10

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
slopes = np.zeros((nMax+1,nMax+1))
slope_errs = np.zeros((nMax+1,nMax+1))
r_sqs = np.zeros((nMax+1,nMax+1))

lnN = [np.log(N) for N in range(1,Nmax+1)]
lnN_r = lnN[100:]
lnSigma = np.zeros((nMax+1,nMax+1,Nmax))
for n1 in range(nMax+1):
    for n2 in range(n1,nMax+1):
        for N in range(1,Nmax+1):
            lnSigma[n1][n2][N-1] = np.log(stdev(overlaps[n1][n2][N-1]))
            #theseSigs = np.zeros((numTrialGroups))
            #for i in range(numTrialGroups):
            #    theseSigs[i] = stdev(overlaps[n1][n2][N-1][trialGroupSize*i:trialGroupSize*(i+1)])
            #lnSigma[n1][n2][N-1] = mean(theseSigs)
        lnSigma_r = lnSigma[n1][n2][100:]
        (slope, intercept, r_sq, slope_err, intercept_err) = regress(lnN_r, lnSigma_r)
        intercepts[n1][n2] = intercept
        intercept_errs[n1][n2] = intercept_err
        slopes[n1][n2] = slope
        slope_errs[n1][n2] = slope_err
        r_sqs[n1][n2] = r_sq

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

    writer.writerow(['slopes'])
    for i in range(len(slopes)):   
        writer.writerow(slopes[i])

    writer.writerow(['slope errors'])
    for i in range(len(slope_errs)):
        writer.writerow(slope_errs[i])

    writer.writerow(['R^2s'])
    for i in range(len(r_sqs)):
        writer.writerow(r_sqs[i])



toc = time.time()
print('runtime (s): '+str(toc-tic))