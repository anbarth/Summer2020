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
numTrialGroups = 10
trialGroupSize = 10

# dimension of discretized position space
D = int((right-left)/dx)

# get all eigenfunctions
eigens = np.zeros((nMax+1,D))
for n in range(nMax+1):
    eigens[n] = shoEigenbra(n,dx,left,right)


### THE MEAT
def findInterceptsOnce():

    overlaps = np.zeros((nMax+1,nMax+1,Nmax,trials))
    for i in range(trialGroupSize):
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
    slopes = np.zeros((nMax+1,nMax+1))

    lnN = [np.log(N) for N in range(1,Nmax+1)]
    lnN_r = lnN[100:]
    lnSigma = np.zeros((nMax+1,nMax+1,Nmax))
    for n1 in range(nMax+1):
        for n2 in range(n1,nMax+1):
            for N in range(1,Nmax+1):
                lnSigma[n1][n2][N-1] = np.log(stdev(overlaps[n1][n2][N-1]))
            lnSigma_r = lnSigma[n1][n2][100:]
            (slope, intercept, r_sq, slope_err, intercept_err) = regress(lnN_r, lnSigma_r)
            intercepts[n1][n2] = intercept
            slopes[n1][n2] = slope
    
    return (intercepts, slopes)

def makeHeatMap():
    intercepts = np.zeros((numTrialGroups,nMax+1,nMax+1))
    slopes = np.zeros((numTrialGroups,nMax+1,nMax+1))
    # TODO this could easily be made parallel processing
    # TODO for the love of god, find a better name than "theseIntercepts" @cs70 smh
    for i in range(numTrialGroups):
        (theseIntercepts, theseSlopes) = findInterceptsOnce()
        intercepts[i] = theseIntercepts
        slopes[i] = theseSlopes

    intercept_avgs = np.zeros((nMax+1,nMax+1))
    intercept_errs = np.zeros((nMax+1,nMax+1))
    slope_avgs = np.zeros((nMax+1,nMax+1))
    slope_errs = np.zeros((nMax+1,nMax+1))

    for n1 in range(nMax+1):
        for n2 in range(n1,nMax+1):
            theseIntercepts = [intercepts[i][n1][n2] for i in range(numTrialGroups)]
            theseSlopes = [slopes[i][n1][n2] for i in range(numTrialGroups)]
            intercept_avg = mean(theseIntercepts)
            intercept_err = stdev(theseIntercepts) / sqrt(numTrialGroups) #TODO to divide or not to divide?
            slope_avg = mean(theseSlopes)
            slope_err = stdev(theseSlopes) / sqrt(theseSlopes)

    with open('heatmap.csv','w') as csvFile:
        writer = csv.writer(csvFile,delimiter=',')

        writer.writerow(['dx='+str(dx)+' over ['+str(left)+','+str(right)+']'])
        writer.writerow(['max N: '+str(Nmax)+', trials: '+str(numTrialGroups)+' groups of '+str(trialGroupSize)])

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
