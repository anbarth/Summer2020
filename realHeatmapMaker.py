#!/usr/bin/env python

import time
import random
import numpy as np
from sho import shoEigenbra,defectEigenstates
from myStats import mean,stdev,regress
import csv
import multiprocessing as mp


### SET UP
nMax = 5 # inclusive
left = -20
right = 20
dx = 0.05
Nmax = 5000
cutoff = 1000 # exclusive


# dimension of discretized position space
D = int((right-left)/dx)

# get all eigenfunctions
depth = 100
width = 1
center = 0
(energies, eigens) = defectEigenstates(depth,width,center,left,right,dx,0,nMax)


### FUNCTION TO BE EXECUTED IN PARALLEL
# makes one ln(sigma) vs ln(N) plot for each (n1,n2), performs one regression per (n1,n2)
def regressOnce():
    sigma = np.zeros((nMax+1,nMax+1,Nmax))
    avg = np.zeros((nMax+1,nMax+1))
    avg2 = np.zeros((nMax+1,nMax+1))

    for N in range(1,Nmax+1):
        zeta = [random.choice([-1,1]) for x in range(D)] # <z|
        psizeta=np.dot(psi1, zeta) # <psi_n1|z>
        zetapsi=np.dot(psi2, zeta) # <z|psi_n2>
        err = psizeta * zetapsi
        if n1 == n2:
            err = err-1

        for n1 in range(nMax+1):
            for n2 in range(n1,nMax+1):
                avg[n1][n2] = (avg[n1][n2] * (N-1) + err) * 1.0/N
                avg2[n1][n2] = (avg2[n1][n2] * (N-1) + err*err) *  1.0/N
                sigma[n1][n2][N-1] = np.sqrt( (avg2 - avg*avg) * 1.0/N )
        
    slopes = np.zeros((nMax+1,nMax+1))
    intercepts = np.zeros((nMax+1,nMax+1))

    for n1 in range(nMax+1):
        for n2 in range(n1,nMax+1):
            lnN = [np.log(N) for N in range(cutoff+1,Nmax+1)]
            lnSigma = [np.log(x) for x in sigma[n1][n2][cutoff:]]

            (slope, intercept, r_sq, slope_err, intercept_err) = myStats.regress(lnN, lnSigma)

            slopes[n1][n2] = slope
            intercepts[n1][n2] = intercept

            if n1 != n2:
                slopes[n2][n1] = slope
                intercepts[n2][n1] = intercept

    avgIntercept = np.sum(intercepts) / ( (nMax+1) * (nMax+1) )
    intercepts = intercepts - avgIntercept

    return (intercepts, slopes, avg)

### HEATMAP MAKING FUNCTION
def makeHeatMap():

    # TODO for the love of god, find a better name than "theseIntercepts" @cs70 smh
    # perform several regressions in parallel
    pool = mp.Pool(mp.cpu_count())
    regression_results = [pool.apply_async(regressOnce,args=[]) for i in range(numRegressions)]
    pool.close()
    pool.join()

    # collect the intercepts & slopes of those regressions
    print(len(regression_results))
    intercepts = [r.get()[0] for r in regression_results]
    slopes = [r.get()[1] for r in regression_results]
    avgOverlaps = [r.get()[2] for r in regression_results]

    # find the avg & std err of the intercepts & slopes
    intercept_avgs = np.zeros((nMax+1,nMax+1))
    intercept_errs = np.zeros((nMax+1,nMax+1))
    slope_avgs = np.zeros((nMax+1,nMax+1))
    slope_errs = np.zeros((nMax+1,nMax+1))
    overlap_avgs = np.zeros((nMax+1,nMax+1))
    overlap_errs = np.zeros((nMax+1,nMax+1))

    for n1 in range(nMax+1):
        for n2 in range(n1,nMax+1):
            theseIntercepts = [intercepts[i][n1][n2] for i in range(numRegressions)]
            theseSlopes = [slopes[i][n1][n2] for i in range(numRegressions)]
            theseOverlaps = [avgOverlaps[i][n1][n2] for i in range(numRegressions)]
            
            intercept_avgs[n1][n2] = mean(theseIntercepts)
            intercept_errs[n1][n2] = stdev(theseIntercepts) / np.sqrt(numRegressions) #TODO to divide or not to divide?
            
            slope_avgs[n1][n2] = mean(theseSlopes)
            slope_errs[n1][n2] = stdev(theseSlopes) / np.sqrt(numRegressions)

            overlap_avgs[n1][n2] = mean(theseOverlaps)
            overlap_errs[n1][n2] = stdev(theseOverlaps) / np.sqrt(numRegressions)

    # write heatmap numbers to csv
    with open('theheatmap.csv','w') as csvFile:
        writer = csv.writer(csvFile,delimiter=',')
        writer.writerow(['dx='+str(dx)+' over ['+str(left)+','+str(right)+']'])
        writer.writerow(['max N: '+str(Nmax)+', trials: '+str(numRegressions)+' groups of '+str(a)+" to "+str(b)])

        writer.writerow(['intercepts'])
        for i in range(len(intercept_avgs)):
            writer.writerow(intercept_avgs[i])

        writer.writerow(['intercept errors'])
        for i in range(len(intercept_errs)):
            writer.writerow(intercept_errs[i])

        writer.writerow(['slopes'])
        for i in range(len(slope_avgs)):   
            writer.writerow(slope_avgs[i])

        writer.writerow(['slope errors'])
        for i in range(len(slope_errs)):
            writer.writerow(slope_errs[i])

        writer.writerow(['overlaps'])
        for i in range(len(overlap_avgs)):
            writer.writerow(overlap_avgs[i])

        writer.writerow(['overlap errors'])
        for i in range(len(overlap_errs)):
            writer.writerow(overlap_errs[i])

random.seed()
tic = time.time()
makeHeatMap()
toc = time.time()
print("runtime (s): "+str(toc-tic))
