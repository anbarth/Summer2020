#!/usr/bin/env python

import time
import random
import numpy as np
from sho import shoEigenbra,defectEigenstates
from myStats import mean,stdev,regress
import csv
import multiprocessing as mp


### SET UP
n1 = 1
n2 = 2
left = -20
right = 20
dx = 0.05
Nmax = 5000
# numRegressions = 1
trialsPerRegression = 50
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
(energies, eigens) = defectEigenstates(depth,width,center,left,right,dx,0,max(n1,n2))
psi1 = eigens[n1]
psi2 = eigens[n2]

### FUNCTION TO BE EXECUTED IN PARALLEL
# makes one ln(sigma) vs ln(N) plot for each (n1,n2), performs one regression per (n1,n2)
def regressOnce():
    overlaps = np.zeros((Nmax,b))
    # run trials, aka, generate different sets of N random vectors
    for i in range(trialsPerRegression):
        # generate Nmax vectors
        for N in range(1,Nmax+1):
            zeta = [random.choice([-1,1]) for x in range(D)] # <z|
            psi1zeta[n] = np.dot(psi1, zeta) # <psi_1|z>
            psi2zeta[n] = np.dot(psi2, zeta) # <psi_2|z>
         
            # <psi_n1|psi_n2>  ~ sum over i( <psi_n1|z_i><z_i|psi_n2> ) / N
            if N == 1:
                overlaps[N-1][i] = np.vdot(psi1zeta, psi2zeta)
            else:
                overlaps[N-1][i] = ( overlaps[N-2][i]*(N-1) + np.vdot(psi1zeta, psi2zeta) ) * 1.0/N



    lnN = [np.log(N) for N in range(100,Nmax+1)]
    #lnN_r = lnN[100:]
    lnSigma = np.zeros((Nmax+1-100))

    for N in range(100,Nmax+1):
        lnSigma[N-100] = np.log(stdev(overlaps[N-100][0:numTrials]))
    (slope, intercept, r_sq, slope_err, intercept_err) = regress(lnN, lnSigma[n1][n2])


    avgOverlap = mean(overlaps[Nmax-1])
    
    ### WRITE OUTPUT
    with open('n1n2.csv','w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',')

        # write specs abt this run
        writer.writerow(['n1='+str(n1)+', n2='+str(n2)+'. dx='+str(dx)+' over ['+str(left)+','+str(right)+']'])
        writer.writerow(['Nmax: '+str(Nmax)+', '+str(numTrials)+' trials'])

        # write slope and intercept
        writer.writerow(['slope',slope,'slope err',slope_err])
        writer.writerow(['incpt',intercept,'incpt err',intercept_err])


        # write ln sigma vs ln N
        writer.writerow(['ln N','ln sigma '])
        for i in range(Nmax-100):
            writer.writerow([lnN_r[i],lnSigma[i]])


random.seed()
tic = time.time()
makeHeatMap()
toc = time.time()
print("runtime (s): "+str(toc-tic))
