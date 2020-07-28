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
n1 = 1
n2 = 2
left = -5
right = 5
dx = 1
Nmax = 150
#numRegressions = 50
#trialsPerRegression = 1000

# dimension of discretized position space
D = int((right-left)/dx)

# get all eigenfunctions
depth = 0
width = 0
center = 0
(energies, eigens) = defectEigenstates(depth,width,center,left,right,dx,0,max(n1,n2))
psi1 = eigens[n1]
psi2 = eigens[n2]

# makes one ln(sigma) vs ln(N) plot 
def regressOnce():
    overlaps = np.zeros((Nmax))
    overlaps_b = np.zeros((Nmax))
    # run trials, aka, generate different sets of N random vectors

    # generate Nmax random vectors
    for N in range(1,Nmax+1):
        zeta = [random.choice([-1,1]) for x in range(D)] # <z|

        psi1zeta = np.dot(psi1,zeta)
        psi2zeta = np.dot(psi2,zeta)

        # <psi_n1|psi_n2>  ~ sum over i( <psi_n1|z_i><z_i|psi_n2> ) / N
        overlaps_b[N-1] = np.vdot(psi1zeta,psi2zeta)
        if N == 1:
            overlaps[N-1] = np.vdot(psi1zeta,psi2zeta)
        else:
            overlaps[N-1] = ( overlaps[N-2]*(N-1) + np.vdot(psi1zeta,psi2zeta) ) * 1.0/N



    lnN = [np.log(N) for N in range(1,Nmax+1)]
    lnN_r = lnN[100:]
    lnSigma = np.zeros((Nmax-100))
    lnSigma_b = np.zeros((Nmax-100))

    for N in range(100,Nmax):
        lnSigma[N-100] = np.log(stdev(overlaps[0:N+1]))
        lnSigma_b[N-100] = np.log(stdev(overlaps_b[0:N+1]))
    #lnSigma_r = lnSigma[n1][n2][100:]
    #print(len(lnN_r),' ',len(lnSigma))
    (slope, intercept, r_sq, slope_err, intercept_err) = regress(lnN_r, lnSigma)
    (slope_b,intercept_b,r_sq_b,slope_berr,intercept_berr) = regress(lnN_r, lnSigma_b)

    avgOverlap = overlaps[Nmax-1]


    ### WRITE OUTPUT
    with open('n1n2.csv','w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',')

        # write specs abt this run
        writer.writerow(['n1='+str(n1)+', n2='+str(n2)+'. dx='+str(dx)+' over ['+str(left)+','+str(right)+']'])
        writer.writerow(['Nmax: '+str(Nmax)])

        # write slope and intercept
        writer.writerow(['slope',slope,'slope err',slope_err])
        writer.writerow(['incpt',intercept,'incpt err',intercept_err])
        writer.writerow(['slope b',slope_b,'slope b err',slope_berr])
        writer.writerow(['incpt b',intercept_b,'incpt b err',intercept_berr])

        # write ln sigma vs ln N
        writer.writerow(['ln N','ln sigma A','ln sigma B'])
        for i in range(Nmax-100):
            writer.writerow([lnN_r[i],lnSigma[i],lnSigma_b[i]])

random.seed()
tic = time.time()
regressOnce()
toc = time.time()
print("runtime (s): "+str(toc-tic))
