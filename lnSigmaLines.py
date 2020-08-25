import time
import random
import numpy as np
import sho
import myStats
from linReg import regress
import matplotlib.pyplot as plt
import importlib as imp
imp.reload(myStats)
imp.reload(sho)

tic = time.time()
random.seed()

# this script produces ln sigma vs ln N plots

###### SET UP ##########
# pairs of states to take inner products between
nPair1 = (0,0)
nPair2 = (1,3)
nPair3 = (10,15)
nPairs = [nPair1,nPair2,nPair3]


left = -5 # bounds of position space
right = 5 # bounds of position space
dx = 0.2 # mesh size to store wavefunctions on

Nmax = 10000 # number of samples
cutoff = 1000 # lowest value of N to include in the linear regression (exclusive)

# dimension of discretized position space
D = int((right-left)/dx)

# get all eigenfunctions
# note that dx_solve and all the defect parameters are hardcoded in here!
# they dont appear elsewhere in the code, so just changing the values here will be fine
(E,psi) = sho.defectEigenstates(500,1,0,left,right,0,max(nPair1+nPair2+nPair3),dx,0.001)


###### COMPUTE INNER PRODUCTS ######
# for each pair of states, keep track of...
sigma = np.zeros((len(nPairs),Nmax)) # standard error, at every step
avg = np.zeros((len(nPairs))) # running average
avg2 = np.zeros((len(nPairs))) # running average^2

# take samples!
for N in range(1,Nmax+1):
    zeta = [random.choice([-1,1]) for x in range(D)] # <z|
    for i in range(len(nPairs)):

        psizeta = np.vdot(psi[nPairs[i][0]], zeta) # <n1|z>
        zetapsi = np.vdot(psi[nPairs[i][1]], zeta) # <z|n2>
        err = psizeta * zetapsi

        # subtract 1 on the diagonal
        if nPairs[i][0] == nPairs[i][1]:
            err = err-1

        avg[i] = (avg[i] * (N-1) + err) * 1.0/N
        avg2[i] = (avg2[i] * (N-1) + err*err) *  1.0/N
        sigma[i][N-1] = np.sqrt( (avg2[i] - avg[i]*avg[i]) * 1.0/N )
    

# get ready to plot
plt.clf()
label = ''

Ns = [N for N in range(3,Nmax+1)]
lnN = [np.log(N) for N in range(cutoff,Nmax+1)]


# go through all the pairs of states
for i in range(len(nPairs)):
    # calculate ln(sigma)
    lnSigma = [np.log(x) for x in sigma[i][cutoff-1:]]

    # regress
    (slope, intercept, r_sq, slope_err, intercept_err) = myStats.regress(lnN, lnSigma)
    intS = str(int(intercept*100000)/100000)
    slopeS = str(int(-1*slope*100000)/100000)
    label += str(nPairs[i])+': '+intS+' - '+slopeS+' x\n'
    
    # plot
    plt.plot(Ns,sigma[i][2:])


# finish up the plot
print(label)
plt.title('max N: '+str(Nmax)+'\ndx='+str(dx)+' over ['+str(left)+','+str(right)+']')
plt.legend([str(nPair1),str(nPair2),str(nPair3)])
plt.xscale('log')
plt.yscale('log')
plt.figtext(0.2,0.2,label)




toc = time.time()
print('runtime (s): '+str(toc-tic))
plt.show()