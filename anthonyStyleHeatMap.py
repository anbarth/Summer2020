#!/usr/bin/env python

import time
import random
import numpy as np
from sho import shoEigenbra,defectEigenstates
from myStats import mean,stdev
from linReg import regress
import csv
import multiprocessing as mp


### SET UP
nMax = 10 # inclusive
left = -20
right = 20
dx = 0.05
Nmax = 5000
numRegressions = 50
#trialsPerRegression = 1000

# dimension of discretized position space
D = int((right-left)/dx)

# get all eigenfunctions
depth = 0
width = 0
center = 0
(energies, eigens) = defectEigenstates(depth,width,center,left,right,dx,0,nMax)

### FUNCTION TO BE EXECUTED IN PARALLEL
# makes one ln(sigma) vs ln(N) plot for each (n1,n2), performs one regression per (n1,n2)
def regressOnce():
    print('regressOnce')
    overlaps = np.zeros((nMax+1,nMax+1,Nmax))
    overlaps_b = np.zeros((nMax+1,nMax+1,Nmax))
    # run trials, aka, generate different sets of N random vectors

    # generate Nmax random vectors
    for N in range(1,Nmax+1):
        zeta = [random.choice([-1,1]) for x in range(D)] # <z|
        psizeta = np.zeros(nMax+1)
        for n in range(nMax+1):
            # TODO some complex conjugate nonesense might be needed here
            psizeta[n] = np.dot(eigens[n], zeta) # <psi_n|z>
        for n1 in range(nMax+1):
            for n2 in range(n1,nMax+1):
                # <psi_n1|psi_n2>  ~ sum over i( <psi_n1|z_i><z_i|psi_n2> ) / N
                overlaps_b[n1][n2][N-1] = np.vdot(psizeta[n1],psizeta[n2])
                if N == 1:
                    overlaps[n1][n2][N-1] = np.vdot(psizeta[n1], psizeta[n2])
                else:
                    overlaps[n1][n2][N-1] = ( overlaps[n1][n2][N-2]*(N-1) + np.vdot(psizeta[n1], psizeta[n2]) ) * 1.0/N

    intercepts = np.zeros((nMax+1,nMax+1))
    slopes = np.zeros((nMax+1,nMax+1))
    intercepts_b = np.zeros((nMax+1,nMax+1))
    slopes_b = np.zeros((nMax+1,nMax+1))
    

    lnN = [np.log(N) for N in range(1,Nmax+1)]
    lnN_r = lnN[100:]
    lnSigma = np.zeros((nMax+1,nMax+1,Nmax-100))
    lnSigma_b = np.zeros((nMax+1,nMax+1,Nmax-100))
    for n1 in range(nMax+1):
        for n2 in range(n1,nMax+1):
            for N in range(100,Nmax):
                lnSigma[n1][n2][N-100] = np.log(stdev(overlaps[n1][n2][0:N+1]))
                lnSigma_b[n1][n2][N-100] = np.log(stdev(overlaps_b[n1][n2][0:N+1]))
            #lnSigma_r = lnSigma[n1][n2][100:]
            #print(len(lnN_r),' ',len(lnSigma))
            (slope, intercept, r_sq, slope_err, intercept_err) = regress(lnN_r, lnSigma[n1][n2])
            (slope_b,intercept_b,garbage,trash,laji) = regress(lnN_r, lnSigma_b[n1][n2])
            intercepts[n1][n2] = intercept
            slopes[n1][n2] = slope
            intercepts_b[n1][n2] = intercept_b
            slopes_b[n1][n2] = slope_b
    
    avgOverlaps = np.zeros((nMax+1,nMax+1))
    for n1 in range(nMax+1):
        for n2 in range(n1,nMax+1):
            avgOverlaps[n1][n2] = overlaps[n1][n2][Nmax-1]


    return (intercepts, slopes,avgOverlaps,intercepts_b,slopes_b)

### HEATMAP MAKING FUNCTION
def makeHeatMap():
    intercepts = np.zeros((numRegressions,nMax+1,nMax+1))
    slopes = np.zeros((numRegressions,nMax+1,nMax+1))
    # TODO for the love of god, find a better name than "theseIntercepts" @cs70 smh
    # perform several regressions in parallel
    pool = mp.Pool(mp.cpu_count())
    regression_results = [pool.apply_async(regressOnce,args=[]) for i in range(numRegressions)]
    pool.close()
    pool.join()
    # collect the intercepts & slopes of those regressions
    intercepts = [r.get()[0] for r in regression_results]
    slopes = [r.get()[1] for r in regression_results]

    intercepts_b = [r.get()[3] for r in regression_results]
    slopes_b = [r.get()[4] for r in regression_results]
    overlaps = [r.get()[2] for r in regression_results]
    
    # find the avg & std err of the intercepts & slopes
    intercept_avgs = np.zeros((nMax+1,nMax+1))
    intercept_errs = np.zeros((nMax+1,nMax+1))
    slope_avgs = np.zeros((nMax+1,nMax+1))
    slope_errs = np.zeros((nMax+1,nMax+1))
    overlap_avgs = np.zeros((nMax+1,nMax+1))
    for n1 in range(nMax+1):
        for n2 in range(n1,nMax+1):
            theseIntercepts = [intercepts[i][n1][n2] for i in range(numRegressions)]
            theseSlopes = [slopes[i][n1][n2] for i in range(numRegressions)]
            theseOverlaps = [overlaps[i][n1][n2] for i in range(numRegressions)]
            intercept_avgs[n1][n2] = mean(theseIntercepts)
            intercept_errs[n1][n2] = stdev(theseIntercepts) / np.sqrt(numRegressions) #TODO to divide or not to divide?
            slope_avgs[n1][n2] = mean(theseSlopes)
            slope_errs[n1][n2] = stdev(theseSlopes) / np.sqrt(numRegressions)
            overlap_avgs[n1][n2] = mean(theseOverlaps)
    # find the avg & std err of the intercepts & slopes
    intercept_bavgs = np.zeros((nMax+1,nMax+1))
    intercept_berrs = np.zeros((nMax+1,nMax+1))
    slope_bavgs = np.zeros((nMax+1,nMax+1))
    slope_berrs = np.zeros((nMax+1,nMax+1))
    for n1 in range(nMax+1):
        for n2 in range(n1,nMax+1):
            theseIntercepts = [intercepts_b[i][n1][n2] for i in range(numRegressions)]
            theseSlopes = [slopes_b[i][n1][n2] for i in range(numRegressions)]
            intercept_bavgs[n1][n2] = mean(theseIntercepts)
            intercept_berrs[n1][n2] = stdev(theseIntercepts) / np.sqrt(numRegressions) #TODO to divide or not to divide?
            slope_bavgs[n1][n2] = mean(theseSlopes)
            slope_berrs[n1][n2] = stdev(theseSlopes) / np.sqrt(numRegressions)
    
    # write heatmap numbers to csv
    with open('anthonyheatmap.csv','w') as csvFile:
        writer = csv.writer(csvFile,delimiter=',')
        writer.writerow(['dx='+str(dx)+' over ['+str(left)+','+str(right)+']'])
        writer.writerow(['max N: '+str(Nmax)+', trials: '+str(numRegressions)])

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

        writer.writerow(['intercepts_b'])
        for i in range(len(intercept_bavgs)):
            writer.writerow(intercept_bavgs[i])

        writer.writerow(['intercept errors_b'])
        for i in range(len(intercept_berrs)):
            writer.writerow(intercept_berrs[i])

        writer.writerow(['slopes_b'])
        for i in range(len(slope_bavgs)):   
            writer.writerow(slope_bavgs[i])

        writer.writerow(['slope errors_b'])
        for i in range(len(slope_berrs)):
            writer.writerow(slope_berrs[i])

        writer.writerow(['overlaps'])
        for i in range(len(overlaps_avgs)):
            writer.writerow(overlap_avgs[i])

random.seed()
tic = time.time()
makeHeatMap()
#regressOnce()
toc = time.time()
print("runtime (s): "+str(toc-tic))
