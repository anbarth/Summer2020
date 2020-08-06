#!/usr/bin/env python3

import time
import random
import numpy as np
import sho
import myStats
import csv
import multiprocessing as mp
import importlib

importlib.reload(sho)

### SET UP
nMax = 20 # inclusive
left = -20
right = 20
dx = 0.05
Nmax = 10000
cutoff = 1000 # exclusive
numRegressions = 100

# dimension of discretized position space
D = int((right-left)/dx)




### FUNCTION TO BE EXECUTED IN PARALLEL
# makes one ln(sigma) vs ln(N) plot for each (n1,n2), performs one regression per (n1,n2)
def regressOnce(eigens):
    #print('regress once')

    sigma = np.zeros((nMax+1,nMax+1,Nmax))
    avg = np.zeros((nMax+1,nMax+1))
    avg2 = np.zeros((nMax+1,nMax+1))

    for N in range(1,Nmax+1):
        zeta = [random.choice([-1,1]) for x in range(D)] # <z|

        for n1 in range(nMax+1):
            # TODO complex conjugate nonsense
            psizeta=np.dot(eigens[n1], zeta) # <psi|z>
            for n2 in range(n1,nMax+1):
                zetaphi=np.dot(eigens[n2], zeta) # <z|phi>
                err = psizeta * zetaphi # <psi|zeta><zeta|phi>
                if n1 == n2:
                    err = err-1
                avg[n1][n2] = (avg[n1][n2] * (N-1) + err) * 1.0/N
                avg2[n1][n2] = (avg2[n1][n2] * (N-1) + err*err) *  1.0/N
                sigma[n1][n2][N-1] = np.sqrt( (avg2[n1][n2] - avg[n1][n2]*avg[n1][n2]) * 1.0/N )
        
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
def makeHeatMap(fname,depth,width,center):

    # get all eigenfunctions
    (energies, eigens) = sho.defectEigenstates(depth,width,center,left,right,dx,0,nMax,0.001)

    # TODO for the love of god, find a better name than "theseIntercepts" @cs70 smh
    # perform several regressions in parallel
    pool = mp.Pool(mp.cpu_count())
    regression_results = [pool.apply_async(regressOnce,args=[eigens]) for i in range(numRegressions)]
    pool.close()
    pool.join()

    # collect the intercepts & slopes of those regressions
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
            
            intercept_avgs[n1][n2] = myStats.mean(theseIntercepts)
            intercept_errs[n1][n2] = myStats.stdev(theseIntercepts) / np.sqrt(numRegressions) #TODO to divide or not to divide?
            
            slope_avgs[n1][n2] = myStats.mean(theseSlopes)
            slope_errs[n1][n2] = myStats.stdev(theseSlopes) / np.sqrt(numRegressions)

            overlap_avgs[n1][n2] = myStats.mean(theseOverlaps)
            overlap_errs[n1][n2] = myStats.stdev(theseOverlaps) / np.sqrt(numRegressions)

    # write heatmap numbers to csv
    with open(fname,'w') as csvFile:
        writer = csv.writer(csvFile,delimiter=',')
        writer.writerow(['dx='+str(dx)+' over ['+str(left)+','+str(right)+']',str(depth),str(width),str(center)])
        writer.writerow(['max N: '+str(Nmax)+', cutoff: '+str(cutoff)+', trials: '+str(numRegressions)])

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
makeHeatMap('run16.csv',10,1,0)
toc = time.time()
print("runtime (s): "+str(toc-tic))

makeHeatMap('run17.csv',100,1,0)
tic = time.time()
print("runtime (s): "+str(tic-toc))

makeHeatMap('run18.csv',1000,1,0)
toc = time.time()
print("runtime (s): "+str(toc-tic))

makeHeatMap('run19.csv',10,1,-1)
tic = time.time()
print("runtime (s): "+str(tic-toc))

makeHeatMap('run20.csv',10,1,1)
toc = time.time()
print("runtime (s): "+str(toc-tic))

#makeHeatMap('run12.csv',10,7.5,0)
#tic = time.time()
#print("runtime (s): "+str(tic-toc))
