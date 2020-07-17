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
nMax = 2 # inclusive
left = -10
right = 10
dx = 2
Nlist = [50,150,500]
sampleSize = 10
trials = 3

# dimension of discretized position space
D = int((right-left)/dx)

# get all eigenfunctions
eigens = np.zeros((nMax+1,D))
for n in range(nMax+1):
    eigens[n] = shoEigenbra(n,dx,left,right)

### PARALLEL SHIT??

def calcOverlaps(N):
    overlaps = np.zeros((nMax+1,nMax+1))
    psizeta = np.zeros((nMax+1,N))
    # pick N random vectors
    for k in range(N):
        zeta = [random.choice([-1,1]) for x in range(D)] # <z|
        for n in range(nMax+1):
            # TODO some complex conjugate nonesense might be needed here
            psizeta[n][k] = np.dot(eigens[n], zeta) # <psi_n|z>
        
    # go over all n1, n2
    for n1 in range(nMax+1):
        for n2 in range(n1, nMax+1):
            # store <phi|psi> = sum <phi|z><z|psi> / N
            overlaps[n1][n2] = np.vdot(psizeta[n1], psizeta[n2])*(1.0/N) 
    
    return overlaps


### THE MEAT

avgSig = np.zeros((len(Nlist),nMax+1,nMax+1))
avgSig_err = np.zeros((len(Nlist),nMax+1,nMax+1))
intercepts = np.zeros((nMax+1,nMax+1))
intercept_errs = np.zeros((nMax+1,nMax+1))

# get aaaaaaaaaall the dataaaaaaaaa
for N_index in range(len(Nlist)):
    N = Nlist[N_index]
    sigmas = np.zeros((nMax+1,nMax+1,trials))

    for i in range(trials):
        # big ol' array for storing all them overlaps
        # TODO the array doesnt technically need to be this big, i only need a triangle
        pool = mp.Pool(mp.cpu_count())
        #overlaps = [pool.apply(calcOverlaps,args=[N]) for j in range(sampleSize)]
        overlaps_results = [pool.apply_async(calcOverlaps,args=[N]) for j in range(sampleSize)]
        pool.close()
        pool.join()
        overlaps = [r.get() for r in overlaps_results]
        # ok, overlaps array is filled in; now put data in sigmas
        for n1 in range(nMax+1):
            for n2 in range(n1,nMax+1):
                sigmas[n1][n2][i] = stdev(overlaps[n1][n2])
    
    # ok, now i have all the sigmas. find avgs & std errs
    for n1 in range(nMax+1):
        for n2 in range(n1,nMax+1):
            avgSig[N_index][n1][n2] = mean(sigmas[n1][n2])
            avgSig_err[N_index][n1][n2] = stdev(sigmas[n1][n2]) / np.sqrt(trials)

# ok, now i have all the data i need to make a plot for every (n1,n2)
lnN = [np.log(N) for N in Nlist]
for n1 in range(nMax+1):
    for n2 in range(n1,nMax+1):
        # here's the data i wanna work with
        lnSigma = [np.log(avgSig[x][n1][n2]) for x in range(len(Nlist))]
        lnSigma_err = [ avgSig_err[x][n1][n2] / avgSig[x][n1][n2] for x in range(len(Nlist))]

        #plt.plot(lnN,lnSigma)
        #plt.show()
        # regress!
        (slope, intercept, r_sq, slope_err, intercept_err) = regress(lnN, lnSigma)
        intercepts[n1][n2] = intercept
        intercept_errs[n1][n2] = intercept_err

# write everything to a csv
with open('heatmap.csv','w') as csvFile:
#with open('heatmap.csv','w',newline='') as csvFile:
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

