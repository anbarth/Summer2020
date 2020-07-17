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

### SET UP

random.seed()
n1 = 2
n2 = 5
left = -20
right = 20
dx = 0.05
Nlist = [10,25,50,150,500,1000]
sampleSize = 50
trials = 10

# dimension of discretized position space
D = int((right-left)/dx)

# get all eigenfunctions
eigens = np.zeros((2,D))
eigens[0] = shoEigenbra(n1,dx,left,right)
eigens[1] = shoEigenbra(n2,dx,left,right)

### PARALLEL SHIT??

def calcOverlap(N):
    #overlaps = np.zeros((nMax+1,nMax+1))
    psizeta = np.zeros((2,N))
    # pick N random vectors
    for k in range(N):
        zeta = [random.choice([-1,1]) for x in range(D)] # <z|
        # TODO some complex conjugate nonesense might be needed here
        psizeta[0][k] = np.dot(eigens[0], zeta) # <psi_n|z>
        psizeta[1][k] = np.dot(eigens[1], zeta) # <psi_n|z>
        
    return np.vdot(psizeta[0], psizeta[1])*(1.0/N) 


### THE MEAT

avgSig = np.zeros((len(Nlist),1))
avgSig_err = np.zeros((len(Nlist),1))
#intercepts = np.zeros((nMax+1,nMax+1))
#intercept_errs = np.zeros((nMax+1,nMax+1))

# get aaaaaaaaaall the dataaaaaaaaa
for N_index in range(len(Nlist)):
    N = Nlist[N_index]
    sigmas = np.zeros((1,trials))
    avgOverlaps = []
    for i in range(trials):
        
        pool = mp.Pool(mp.cpu_count())
        overlaps_results = [pool.apply_async(calcOverlap,args=[N]) for j in range(sampleSize)]
        pool.close()
        pool.join()
        overlaps = [r.get() for r in overlaps_results]
        # ok, overlaps array is filled in; now put data in sigmas
        avgOverlaps.append(mean(overlaps))
        sigmas[0][i] = stdev(overlaps)
    
    # ok, now i have all the sigmas. find avgs & std errs
    avgSig[N_index] = mean(sigmas[0])
    avgSig_err[N_index] = stdev(sigmas[0]) / np.sqrt(trials)
    print(mean(avgOverlaps))
# ok, now i have all the data i need to make a plot for every (n1,n2)
lnN = [np.log(N) for N in Nlist]

# here's the data i wanna work with
lnSigma = [np.log(avgSig[x]) for x in range(len(Nlist))]
lnSigma_err = [avgSig_err[x] / avgSig[x] for x in range(len(Nlist))]

#plt.plot(lnN,lnSigma)
#plt.show()
# regress!
(slope, intercept, r_sq, slope_err, intercept_err) = regress(lnN, lnSigma)


# write everything to a csv
with open('n1n2.csv','w') as csvFile:
#with open('n1n2.csv','w',newline='') as csvFile:
    writer = csv.writer(csvFile, delimiter=',')

    # write specs abt this run
    writer.writerow(['n1='+str(n1)+', n2='+str(n2)+'. dx='+str(dx)+' over ['+str(left)+','+str(right)+']'])
    writer.writerow(['sample size: '+str(sampleSize)+', '+str(trials)+' trials'])

    # write slope and intercept
    writer.writerow(['slope',slope[0],'slope err',slope_err])
    writer.writerow(['incpt',intercept[0],'incpt err',intercept_err])

    # write ln sigma vs ln N
    writer.writerow(['ln N','ln sigma','ln sigma err'])
    for i in range(len(Nlist)):
        writer.writerow([lnN[i],lnSigma[i][0],lnSigma_err[i][0]])



toc = time.time()
print("runtime (s): "+str(toc-tic))
slopeS = str(int(slope[0]*10000)/10000.0)
slope_errS = str(int(slope_err*10000)/10000.0)
interS = str(int(intercept[0]*1000)/1000.0)
inter_errS = str(int(intercept_err*1000)/1000.0)
print("slope: "+slopeS+" +/- "+slope_errS)
print("intercept: "+interS+" +/- "+inter_errS)

