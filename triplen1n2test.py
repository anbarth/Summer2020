import time
import random
import numpy as np
import sho
import myStats
from linReg import regress
import matplotlib.pyplot as plt
import csv
import multiprocessing as mp
import importlib as imp

imp.reload(myStats)
imp.reload(sho)

tic = time.time()
random.seed()

### SET UP
nPair1 = (2,5)
nPair2 = (2,15)
nPair3 = (15,17)
nPairs = [nPair1,nPair2,nPair3]
left = -20
right = 20
dx = 0.05
Nmax = 10000
#TODO make cutoff a param

# dimension of discretized position space
D = int((right-left)/dx)

# get all eigenfunctions
(E,psi) = sho.defectEigenstates(1000,1,0,left,right,dx,0,max(nPair1+nPair2+nPair3),0.001)

### THE MEAT
sigma = np.zeros((len(nPairs),Nmax))
#sigma2 = np.zeros((Nmax))
#errs = np.zeros((Nmax))
avg = np.zeros((len(nPairs)))
avg2 = np.zeros((len(nPairs)))
#overlaps = np.zeros((len(nPairs),Nmax))
for N in range(1,Nmax+1):
    zeta = [random.choice([-1,1]) for x in range(D)] # <z|
    for i in range(len(nPairs)):

        psizeta=np.dot(psi[nPairs[i][0]], zeta) # <psi_n1|z>
        zetapsi=np.dot(psi[nPairs[i][1]], zeta) # <z|psi_n2>
        err = psizeta * zetapsi
        if nPairs[i][0] == nPairs[i][1]:
            err = err-1

        avg[i] = (avg[i] * (N-1) + err) * 1.0/N
        avg2[i] = (avg2[i] * (N-1) + err*err) *  1.0/N
        #overlaps[i][N-1] = avg[i]
        sigma[i][N-1] = np.sqrt( (avg2[i] - avg[i]*avg[i]) * 1.0/N )
    

lnN = [np.log(N) for N in range(1000,Nmax+1)]
lnSigma = []
plt.clf()
label = ''
for i in range(len(nPairs)):
    lnSigma.append( [np.log(x) for x in sigma[i][999:]] )
    (slope, intercept, r_sq, slope_err, intercept_err) = myStats.regress(lnN, lnSigma[i])
    intS = str(int(intercept*100000)/100000)
    slopeS = str(int(-1*slope*100000)/100000)
    label += str(nPairs[i])+': '+intS+' - '+slopeS+' x\n'
    plt.plot(lnN,lnSigma[i])
    #plt.plot(lnN,overlaps[i][249:])

print(label)
plt.title('max N: '+str(Nmax)+'\ndx='+str(dx)+' over ['+str(left)+','+str(right)+']')
plt.legend([str(nPair1),str(nPair2),str(nPair3)])
plt.figtext(0.2,0.2,label)

#with open('n1n2.csv','w') as csvFile:
with open('n1n2.csv','w',newline='') as csvFile:
    writer = csv.writer(csvFile,delimiter=',')

    #writer.writerow(['n1='+str(n1)+', n2='+str(n2)])
    writer.writerow(['dx='+str(dx)+' over ['+str(left)+','+str(right)+']'])
    writer.writerow(['max N: '+str(Nmax)])


    # write ln sigma vs ln N
    writer.writerow(['ln N','ln sigma 1','ln sigma 2','ln sigma 3'])
    for i in range(len(lnN)):
        writer.writerow([lnN[i],lnSigma[0][i],lnSigma[1][i],lnSigma[2][i]])


toc = time.time()
print('runtime (s): '+str(toc-tic))
plt.show()