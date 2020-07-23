import random
import numpy as np
import csv
import math
import time
import matplotlib.pyplot as plt
import statistics
from sho import shoEigenbra, shoEigenket

# estimates the overlap using N stochastic vectors
# does that many time (specified # of trials)
# shows the histogram of overlap estimations
# see july 1 log for many example outputs!
def makeOverlapsHisto(n1,n2,left,right,dx,N,trials,showGraph=True,timing=True,fname='newfig.png'):
    ### STEP 1: SET UP
    tic = time.perf_counter()
    plt.clf()

    # dimension of discretized position space
    D = int((right-left)/dx)

    # get the appropriate eigenstates
    psi = shoEigenket(n1,dx,left,right)
    phi = shoEigenbra(n1,dx,left,right)
  
    ### STEP 2: CALCULATE OVERLAP MANY TIMES
    overlaps = []
    for j in range(trials):
        # calculate the overlap using N random vectors
        runningTot = 0
        for i in range(N):
            zeta = [[random.choice([-1,1])] for i in range(D)] # |z>
            phizeta = np.matmul(phi, zeta) # <phi|z>
            zetapsi = np.matmul(np.transpose(zeta), psi) # <z|psi>
            prod = phizeta * zetapsi
            runningTot = runningTot + prod
        runningTot = runningTot*(1/N) 
        overlaps.append(runningTot[0][0])

    ### STEP 3: OUTPUT
    if timing:
        toc = time.perf_counter()
        print("runtime "+str(toc-tic))

    if showGraph:
        sigma = int( statistics.stdev(resultsTable)*1000 ) / 1000
        plt.hist(resultsTable, bins='auto')
        plt.figtext(.7,.75,'sigma='+str(sigma))
        plt.savefig(fname)
        #plt.show()

n1=1
n2=1
left=-20
right=20
dx=0.05
N=50
trialsList=[25,50,100,500]

for trials in trialsList:
    for i in range(3):
        filename = 'n50t'+str(trials)+'_'+str(i)+'.png'
        makeOverlapsHisto(n1,n2,left,right,dx,N,trials,fname=filename)