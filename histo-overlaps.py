import random
import numpy as np
import csv
import math
import time
import matplotlib.pyplot as plt
import statistics
import sho
import importlib
importlib.reload(sho)

# this script produces a histogram representing one set of N stochastic vectors, calculating one inner product

def makeOverlapsHisto(n1,n2,left,right,dx,N,fname='newfig.png'):
    ''' takes N samples to estimate the overlap between eigenstates n1 and n2
        position space bounded by [left, right], with mesh size dx
        outputs a histogram of <n1|zeta><zeta|n2> values '''

    ### STEP 1: SET UP
    tic = time.perf_counter()
    plt.clf()

    # dimension of discretized position space
    D = int((right-left)/dx)

    # get the appropriate eigenstates
    # note that dx_solve and all the defect parameters are hardcoded in here!
    # they dont appear elsewhere in the code, so just changing the values here will be fine
    (E,eigen) = sho.defectEigenstates(0,0,0,left,right,min(n1,n2),max(n1,n2),dx,0.001)
    psi = eigen[0]
    phi = eigen[-1]
  
    ### STEP 2: TAKE SAMPLES
    overlaps = []
    for j in range(N):
        zeta = [random.choice([-1,1]) for i in range(D)] # |z>
        phizeta = np.vdot(phi, zeta) # <phi|z>
        zetapsi = np.vdot(zeta, psi) # <z|psi>
        prod = phizeta * zetapsi
        overlaps.append(prod)

    ### STEP 3: OUTPUT
    # print runtime
    toc = time.perf_counter()
    print("runtime "+str(toc-tic))

    # make plot
    avg = int( statistics.mean(overlaps)*1000 ) / 1000
    sigma = int( statistics.stdev(overlaps)*1000 ) / 1000
    plt.hist(overlaps, bins='auto')
    plt.figtext(.7,.75,'avg='+str(avg)+'\nsigma='+str(sigma))
    plt.savefig(fname)
    plt.show()

# parameters
n1=0
n2=1
left=-20
right=20
dx=0.2
N=50000
filename = 'data/aug 14/0x0x0-0-1'

makeOverlapsHisto(n1,n2,left,right,dx,N,fname=filename)