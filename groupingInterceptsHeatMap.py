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
numRegressions = 100
#trialsPerRegression = 100
a = 100
b = 100

# dimension of discretized position space
D = int((right-left)/dx)

# get all eigenfunctions
#eigens = np.zeros((nMax+1,D))
#for n in range(nMax+1):
#    eigens[n] = shoEigenbra(n,dx,left,right)
depth = 100
width = 1
center = 0
(energies, eigens) = defectEigenstates(depth,width,center,left,right,dx,0,nMax)


### FUNCTION TO BE EXECUTED IN PARALLEL
# makes one ln(sigma) vs ln(N) plot for each (n1,n2), performs one regression per (n1,n2)
def regressOnce():
    print("the boys are waiting")
    overlaps = np.zeros((nMax+1,nMax+1,Nmax,b))
    # run trials, aka, generate different sets of N random vectors
    currNmax = Nmax
    for i in range(b):
        while i >= (a-b)*1.0/(Nmax-110)*(currNmax-110)+b:
            currNmax -= 1
        # generate currNmax random vectors
        for N in range(1,currNmax+1):
            zeta = [random.choice([-1,1]) for x in range(D)] # <z|
            psizeta = np.zeros(nMax+1)
          
            for n in range(nMax+1):
                # TODO some complex conjugate nonesense might be needed here
                psizeta[n] = np.dot(eigens[n], zeta) # <psi_n|z>
         
            for n1 in range(nMax+1):
                for n2 in range(n1,nMax+1):
                    # <psi_n1|psi_n2>  ~ sum over i( <psi_n1|z_i><z_i|psi_n2> ) / N
                    if N == 1:
                        overlaps[n1][n2][N-1][i] = np.vdot(psizeta[n1], psizeta[n2])
                    else:
                        overlaps[n1][n2][N-1][i] = ( overlaps[n1][n2][N-2][i]*(N-1) + np.vdot(psizeta[n1], psizeta[n2]) ) * 1.0/N

    intercepts = np.zeros((nMax+1,nMax+1))
    slopes = np.zeros((nMax+1,nMax+1))
    avgOverlaps = np.zeros((nMax+1,nMax+1))

    lnN = [np.log(N) for N in range(100,Nmax+1)]
    #lnN_r = lnN[100:]
    lnSigma = np.zeros((nMax+1,nMax+1,Nmax+1-100))
    for n1 in range(nMax+1):
        for n2 in range(n1,nMax+1):
            for N in range(100,Nmax+1):
                # TODO this is hell
                # TODO also does lnSigma need have an (n1,n2) dimension??
                numTrials = min( b, int(1 + (a-b)*1.0/(Nmax-110)*(N-110)+b) )
                lnSigma[n1][n2][N-100] = np.log(stdev(overlaps[n1][n2][N-100][0:numTrials]))
            (slope, intercept, r_sq, slope_err, intercept_err) = regress(lnN, lnSigma[n1][n2])
            intercepts[n1][n2] = intercept
            slopes[n1][n2] = slope
    for n1 in range(nMax+1):
        for n2 in range(n1,nMax+1):
            # to find the avg overlap, ill use just the trials that went all the way out to Nmax
            # mostly bc im lazy
            # it shouldnt matter
            avgOverlaps[n1][n2] = mean(overlaps[n1][n2][Nmax-1][0:a])
    
    print("my milkshake brings all the boys to the yard")

    return (intercepts, slopes, avgOverlaps)

### HEATMAP MAKING FUNCTION
def makeHeatMap():
    #intercepts = np.zeros((numRegressions,nMax+1,nMax+1))
    #slopes = np.zeros((numRegressions,nMax+1,nMax+1))
    # TODO for the love of god, find a better name than "theseIntercepts" @cs70 smh
    # perform several regressions in parallel
    pool = mp.Pool(mp.cpu_count())
    regression_results = [pool.apply_async(regressOnce,args=[]) for i in range(numRegressions)]
    #intercepts = [r.get()[0] for r in regression_results]
    #slopes = [r.get()[1] for r in regression_results]
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
