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
n1 = 2
n2 = 5
left = -10
right = 10
dx = 2
Nmax = 100
#sampleSize = 50
trials = 10
#numTrialGroups = 10
#trialGroupSize = 10

# dimension of discretized position space
D = int((right-left)/dx)

# get all eigenfunctions
eigens = np.zeros((2,D))
eigens[0] = shoEigenbra(n1,dx,left,right)
eigens[1] = shoEigenbra(n2,dx,left,right)

### THE MEAT
overlaps = np.zeros((Nmax,trials))
for i in range(trials):
    print("trial "+str(i))
    psizeta = np.zeros((2,Nmax))
    for N in range(1,Nmax+1):
        zeta = [random.choice([-1,1]) for x in range(D)] # <z|
        #col = np.zeros((nMax+1))
        psizeta[0][N-1]=(np.dot(eigens[0], zeta)) # <psi_n1|z>
        psizeta[1][N-1]=(np.dot(eigens[1], zeta)) # <psi_n2|z>
        
        overlaps[N-1][i] = np.vdot(psizeta[0], psizeta[1])*(1.0/N)


lnN = [np.log(N) for N in range(1,Nmax+1)]
lnN_r = lnN[100:]
lnSigma = np.zeros((Nmax))

for N in range(1,Nmax+1):
    lnSigma[N-1] = np.log(stdev(overlaps[N-1]))

lnSigma_r = lnSigma[100:]
(slope, intercept, r_sq, slope_err, intercept_err) = regress(lnN, lnSigma)


with open('n1n2.csv','w') as csvFile:
    writer = csv.writer(csvFile,delimiter=',')

    writer.writerow(['n1='+str(n1)+', n2='+str(n2)])
    writer.writerow(['dx='+str(dx)+' over ['+str(left)+','+str(right)+']'])
    writer.writerow(['max N: '+str(Nmax)+', trials: '+str(trials)])

    # write slope and intercept
    writer.writerow(['slope',slope,'slope err',slope_err])
    writer.writerow(['incpt',intercept,'incpt err',intercept_err])

    # write ln sigma vs ln N
    writer.writerow(['ln N','ln sigma'])
    for i in range(len(lnN)):
        writer.writerow([lnN[i],lnSigma[i]])




toc = time.time()
print('runtime (s): '+str(toc-tic))