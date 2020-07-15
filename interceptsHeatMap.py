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
nMax = 5 # inclusive
left = -10
right = 10
dx = 2
Nlist = [50,150,500]
sampleSize = 25
trials = 10

# dimension of discretized position space
D = int((right-left)/dx)

# get all eigenfunctions
eigens = np.zeros((nMax+1,D))
for n in range(nMax+1):
    eigens[n] = shoEigenbra(n,dx,left,right)

avgSig = np.zeros((len(Nlist),nMax+1,nMax+1))
avgSig_err = np.zeros((len(Nlist),nMax+1,nMax+1))
intercepts = np.zeros((nMax+1,nMax+1))
intercept_errs = np.zeros((nMax+1,nMax+1))

# get aaaaaaaaaall the dataaaaaaaaa
for N_index in range(len(Nlist)):
    N = Nlist[N_index]
    sigmas = np.zeros((trials,nMax+1,nMax+1))

    for i in range(trials):
        # big ol' array for storing all them overlaps
        # TODO the array doesnt technically need to be this big, i only need a triangle
        #overlaps = np.zeros((sampleSize,nMax+1,nMax+1))
        for j in range(sampleSize):
            overlaps = np.zeros((nMax+1,nMax+1))
            psizeta = np.zeros((nMax+1,N))
            # pick N random vectors
            for k in range(N):
                zeta = [[random.choice([-1,1])] for x in range(D)] # |z>
                for n in range(nMax+1):
                    psizeta[n][k] = np.matmul(eigens[n], zeta) # <psi_n|z>
                
            # go over all n1, n2
            for n1 in range(nMax+1):
                for n2 in range(n1, nMax+1):
                    sum = np.vdot(psizeta[n1], psizeta[n2])
                    overlap = sum*(1.0/N)
                    overlaps[i][n1][n2] = overlap

        # ok, overlaps array is filled in; now put data in sigmas
        for n1 in range(nMax+1):
            for n2 in range(n1,nMax+1):
                theseOverlaps = [overlaps[x][n1][n2] for x in range(sampleSize)]
                sigmas[i][n1][n2] = stdev(theseOverlaps)
    
    # ok, now i have all the sigmas. find avgs & std errs
    for n1 in range(nMax+1):
        for n2 in range(n1,nMax+1):
            sigs = [sigmas[x][n1][n2] for x in range(trials)]
            avgSig[N_index][n1][n2] = mean(sigs)
            avgSig_err[N_index][n1][n2] = stdev(sigs)

# ok, now i have all the data i need to make a plot for every (n1,n2)
lnN = [np.log(N) for N in Nlist]
for n1 in range(nMax+1):
    for n2 in range(n1,nMax+1):
        # here's the data i wanna work with
        lnSigma = [np.log(avgSig[x][n1][n2]) for x in range(len(Nlist))]
        lnSigma_err = [ avgSig_err[x][n1][n2] / avgSig[x][n1][n2] for x in range(len(Nlist))]

        #plt.plot(lnN,lnSigma)
        # regress!
        (slope, intercept, r_sq, slope_err, intercept_err) = regress(lnN, lnSigma)
        intercepts[n1][n2] = intercept
        intercept_errs[n1][n2] = intercept_err

# write everything to a csv
#with open('heatmap.csv','w') as csvFile:
with open('heatmap.csv','w',newline='') as csvFile:
    writer = csv.writer(csvFile, delimiter=',')

    # write specs abt this run
    writer.writerow(['dx='+str(dx)+' over ['+str(left)+','+str(right)+']'])
    writer.writerow(['sample size: '+str(sampleSize)+', '+str(trials)+' trials'])

    # write intercepts
    writer.writerow(['intercepts'])
    for i in range(len(intercepts)):
        writer.writerow(intercepts[i])

    # write intercept errors
    writer.writerow(['intercept errors'])
    for i in range(len(intercept_errs)):
        writer.writerow(intercept_errs[i])

toc = time.time()
print("runtime (s): "+str(toc-tic))

