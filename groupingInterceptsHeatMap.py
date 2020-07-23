import time
import random
import numpy as np
from sho import shoEigenbra,defectEigenstates
from myStats import mean,stdev
from linReg import regress
#import matplotlib.pyplot as plt
import csv
import multiprocessing as mp





### SET UP
nMax = 5 # inclusive
left = -20
right = 20
dx = 0.05
Nmax = 5000
numRegressions = 100
trialsPerRegression = 1000

# dimension of discretized position space
D = int((right-left)/dx)

# get all eigenfunctions
#eigens = np.zeros((nMax+1,D))
#for n in range(nMax+1):
#    eigens[n] = shoEigenbra(n,dx,left,right)
depth = 15
width = 2
center = 0
(energies, eigens) = defectEigenstates(depth,width,center,left,right,dx,0,nMax)

### FUNCTION TO BE EXECUTED IN PARALLEL
# makes one ln(sigma) vs ln(N) plot for each (n1,n2), performs one regression per (n1,n2)
def regressOnce():
    overlaps = np.zeros((nMax+1,nMax+1,Nmax,trialsPerRegression))
    # run trials, aka, generate different sets of N random vectors
    for i in range(trialsPerRegression):
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
                    if N == 1:
                        overlaps[n1][n2][N-1][i] = np.vdot(psizeta[n1], psizeta[n2])
                    else:
                        overlaps[n1][n2][N-1][i] = ( overlaps[n1][n2][N-2][i]*(N-1) + np.vdot(psizeta[n1], psizeta[n2]) ) * 1.0/N

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

    # find the avg & std err of the intercepts & slopes
    intercept_avgs = np.zeros((nMax+1,nMax+1))
    intercept_errs = np.zeros((nMax+1,nMax+1))
    slope_avgs = np.zeros((nMax+1,nMax+1))
    slope_errs = np.zeros((nMax+1,nMax+1))
    for n1 in range(nMax+1):
        for n2 in range(n1,nMax+1):
            theseIntercepts = [intercepts[i][n1][n2] for i in range(numRegressions)]
            theseSlopes = [slopes[i][n1][n2] for i in range(numRegressions)]
            intercept_avgs[n1][n2] = mean(theseIntercepts)
            intercept_errs[n1][n2] = stdev(theseIntercepts) / np.sqrt(numRegressions) #TODO to divide or not to divide?
            slope_avgs[n1][n2] = mean(theseSlopes)
            slope_errs[n1][n2] = stdev(theseSlopes) / np.sqrt(numRegressions)

    # write heatmap numbers to csv
    with open('bashheatmap.csv','w') as csvFile:
        writer = csv.writer(csvFile,delimiter=',')
        writer.writerow(['dx='+str(dx)+' over ['+str(left)+','+str(right)+']'])
        writer.writerow(['max N: '+str(Nmax)+', trials: '+str(numRegressions)+' groups of '+str(trialsPerRegression)])

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
makeHeatMap()
toc = time.time()
print("runtime (s): "+str(toc-tic))
