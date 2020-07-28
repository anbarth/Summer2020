#!/usr/bin/env python

import time
import random
import numpy as np
from sho import shoEigenbra,defectEigenstates
from myStats import mean,stdev,regress
import csv
import multiprocessing as mp


### SET UP
nMax = 10 # inclusive
left = -20
right = 20
dx = 0.05
Nmax = 5000
numRegressions = 50
#trialsPerRegression = 100
#a = 100
#b = 100

# dimension of discretized position space
D = int((right-left)/dx)

# get all eigenfunctions
#eigens = np.zeros((nMax+1,D))
#for n in range(nMax+1):
#    eigens[n] = shoEigenbra(n,dx,left,right)
depth = 0
width = 0
center = 0
(energies, eigens) = defectEigenstates(depth,width,center,left,right,dx,0,nMax)


### FUNCTION TO BE EXECUTED IN PARALLEL
# makes one ln(sigma) vs ln(N) plot for each (n1,n2), performs one regression per (n1,n2)

def returnSlopeHeat():
    avg = 0.
    avg2 = 0.
    maxOrder = nMax
    maxSamples = Nmax
    #vals = eigens
    stdev = np.zeros((maxOrder+1, maxOrder+1, maxSamples + 1), dtype=float)
    for sample in range(1, maxSamples + 1):
        #print(sample)
        chi = np.random.rand(np.shape(eigens)[1])
        for k in range(np.size(chi)):
            if chi[k] < 0.5: chi[k] = -1.
            if chi[k] >= 0.5: chi[k] = 1.
        for i in range(0, maxOrder+1):
            for j in range(i, maxOrder+1):
                sm = sum(eigens[i] * chi) * sum(eigens[j] * chi)
                if i == j:
                    sm = sm-1
                avg = (avg * float(sample - 1) + sm) / float(sample)
                avg2 = (avg2 * float(sample - 1) + sm ** 2.) / float(sample)
                stdev[i][j][sample] = np.sqrt(abs((avg ** 2) - avg2)) / np.sqrt(float(sample))
    slopes = np.zeros((maxOrder + 1, maxOrder + 1))
    intercepts = np.zeros((maxOrder + 1, maxOrder + 1))

    for i in range(0, maxOrder+1):
        for j in range(i, maxOrder+1):
            domainSamples = range(1, maxSamples + 1)
            slopes[i, j], intercepts[i,j] = np.polyfit(np.log(domainSamples[100:]), np.log(stdev[i][j][101:]), 1)
            if i != j:
                slopes[j,i] = slopes[i,j]
                intercepts[j, i] = intercepts[i, j]
    return slopes, intercepts


### HEATMAP MAKING FUNCTION
def makeHeatMap():
    #intercepts = np.zeros((numRegressions,nMax+1,nMax+1))
    #slopes = np.zeros((numRegressions,nMax+1,nMax+1))
    # TODO for the love of god, find a better name than "theseIntercepts" @cs70 smh
    # perform several regressions in parallel
    pool = mp.Pool(mp.cpu_count())
    regression_results = [pool.apply_async(returnSlopeHeat,args=[]) for i in range(numRegressions)]

    pool.close()
    pool.join()
    # collect the intercepts & slopes of those regressions
    print(len(regression_results))
    intercepts = [r.get()[1] for r in regression_results]
    slopes = [r.get()[0] for r in regression_results]

    # find the avg & std err of the intercepts & slopes
    intercept_avgs = np.zeros((nMax+1,nMax+1))
    intercept_errs = np.zeros((nMax+1,nMax+1))
    slope_avgs = np.zeros((nMax+1,nMax+1))
    slope_errs = np.zeros((nMax+1,nMax+1))


    for n1 in range(nMax+1):
        for n2 in range(n1,nMax+1):
            theseIntercepts = [intercepts[i][n1][n2] for i in range(numRegressions)]
            theseSlopes = [slopes[i][n1][n2] for i in range(numRegressions)]
            #theseOverlaps = [avgOverlaps[i][n1][n2] for i in range(numRegressions)]
            intercept_avgs[n1][n2] = mean(theseIntercepts)
            intercept_errs[n1][n2] = stdev(theseIntercepts) / np.sqrt(numRegressions) #TODO to divide or not to divide?
            slope_avgs[n1][n2] = mean(theseSlopes)
            slope_errs[n1][n2] = stdev(theseSlopes) / np.sqrt(numRegressions)

            #overlap_avgs[n1][n2] = mean(theseOverlaps)
            #overlap_errs[n1][n2] = stdev(theseOverlaps) / np.sqrt(numRegressions)

    # write heatmap numbers to csv
    with open('theheatmap.csv','w') as csvFile:
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


random.seed()
tic = time.time()
returnSlopeHeat()
#makeHeatMap()
toc = time.time()
print("runtime (s): "+str(toc-tic))
