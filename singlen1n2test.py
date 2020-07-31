import time
import random
import numpy as np
from sho import shoEigenbra,defectEigenstates
import myStats
from linReg import regress
import matplotlib.pyplot as plt
import csv
import multiprocessing as mp
import imp

imp.reload(myStats)

tic = time.time()
random.seed()

### SET UP
n1 = 0
n2 = 10
left = -20
right = 20
dx = 0.05
Nmax = 10000
cutoff = 1000 # exclusive value of N

# dimension of discretized position space
D = int((right-left)/dx)

# get all eigenfunctions
depth=100
width=1
center=0
(E,psi) = defectEigenstates(depth,width,center,left,right,dx,min(n1,n2),max(n1,n2))
psi1 = psi[0]
psi2= psi[-1]

### THE MEAT
sigma = np.zeros((Nmax))
#sigma2 = np.zeros((Nmax))
#errs = np.zeros((Nmax))
avg = 0
avg2 = 0
for N in range(1,Nmax+1):
    zeta = [random.choice([-1,1]) for x in range(D)] # <z|
    psizeta=np.dot(psi1, zeta) # <psi_n1|z>
    zetapsi=np.dot(psi2, zeta) # <z|psi_n2>
    err = psizeta * zetapsi
    if n1 == n2:
        err = err-1

    avg = (avg * (N-1) + err) * 1.0/N
    avg2 = (avg2 * (N-1) + err*err) *  1.0/N
    #errs[N-1] = err
    #sigma1[N-1] = myStats.stdev(errs[0:N]) * 1.0 / np.sqrt(N)
    sigma[N-1] = np.sqrt( (avg2 - avg*avg) * 1.0/N )
    

lnN = [np.log(N) for N in range(1,Nmax+1)]
lnSigma = [0] + [np.log(x) for x in sigma[1:]]


(slope, intercept, r_sq, slope_err, intercept_err) = myStats.regress(lnN[cutoff:], lnSigma[cutoff:])
(slope_fx, intercept_fx, r_sq_fx) = myStats.regressFixedSlope(lnN[cutoff:], lnSigma[cutoff:],slope=-0.5)

plt.clf()
plt.axvline(x=np.log(cutoff+1),ymin=0,ymax=1,color='black')
plt.plot(lnN,lnSigma)

lnSig_model = [slope*x + intercept for x in lnN]
lnSig_model_fx = [slope_fx*x + intercept_fx for x in lnN]

plt.plot(lnN,lnSig_model)
plt.plot(lnN,lnSig_model_fx)

label1 = str(int(slope*1000)/1000) + ' x + '+str(int(intercept*1000)/1000)+': R^2 = '+str(int(r_sq*1000)/1000)
label2 = str(int(slope_fx*1000)/1000) + ' x + '+str(int(intercept_fx*1000)/1000)+': R^2 = '+str(int(r_sq_fx*1000)/1000)
plt.figtext(0.2,0.2,label1+'\n'+label2)
plt.title('n1='+str(n1)+', n2='+str(n2)+'\nmax N: '+str(Nmax)+'. dx='+str(dx)+' over ['+str(left)+','+str(right)+']')
plt.legend(['cutoff: '+str(cutoff),'data','model','model (fixed slope)'],loc='upper right')

#with open('n1n2.csv','w') as csvFile:
with open('n1n2.csv','w',newline='') as csvFile:
    writer = csv.writer(csvFile,delimiter=',')

    writer.writerow(['n1='+str(n1)+', n2='+str(n2)])
    writer.writerow(['dx='+str(dx)+' over ['+str(left)+','+str(right)+']'])
    writer.writerow(['max N: '+str(Nmax)])

    # write slope and intercept
    writer.writerow(['slope',slope])
    writer.writerow(['incpt',intercept])

    # write ln sigma vs ln N
    writer.writerow(['ln N','ln sigma'])
    for i in range(len(lnN)):
        writer.writerow([lnN[i],lnSigma[i]])




toc = time.time()
print('runtime (s): '+str(toc-tic))
plt.show()
