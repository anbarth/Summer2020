import random
import numpy as np
import csv
import math
import time
import matplotlib.pyplot as plt

def makeOverlapsHisto(n1,n2,left,right,dx,N,trials,showGraph=True,writeToCsv=True,timing=True):
    ### STEP 1: SET UP
    tic = time.perf_counter()

    # dimension of discretized position space
    D = int((right-left)/dx)

    # construct the phi and psi matrices
    psi = np.zeros((D,1))
    phi = np.zeros((1,D))

    # make hermite polynomial objects
    n1_arr = [0]*(n1+1)
    n1_arr[-1] = 1
    n2_arr = [0]*(n2+1)
    n2_arr[-1] = 1
    herm1 = np.polynomial.hermite.Hermite(n1_arr,[left,right])
    herm2 = np.polynomial.hermite.Hermite(n2_arr,[left,right])
    herm1_arr = herm1.linspace(n=D)[1]
    herm2_arr = herm2.linspace(n=D)[1]

    # psi and phi's norm-squareds, so i can normalize later
    norm1 = 0
    norm2 = 0

    x = left
    for i in range(D):
        psi[i][0] = math.exp(-1*x*x)*herm1_arr[i]
        phi[0][i] = math.exp(-1*x*x)*herm2_arr[i]
        norm1 += psi[i][0]*psi[i][0]
        norm2 += phi[0][i]*phi[0][i]
        x += dx
    # normalize
    psi = psi*(1/math.sqrt(norm1))
    phi = phi*(1/math.sqrt(norm2))

    ### STEP 2: RUN TRIALS
    resultsTable = []

    for j in range(trials):
        runningTot = 0
        for i in range(N):
            zeta = [[random.choice([-1,1])] for i in range(D)]
            phizeta = np.matmul(phi, zeta) # <phi|z>
            zetapsi = np.matmul(np.transpose(zeta), psi) # <z|psi>
            prod = phizeta * zetapsi
            runningTot = runningTot + prod

        runningTot = runningTot*(1/N) 
        resultsTable.append(runningTot[0][0])
    



    ### STEP 4: OUTPUT
    if writeToCsv:

        with open('overlaps-histo.csv','w',newline='') as csvFile:
            writer = csv.writer(csvFile, delimiter=',')

            # write specs abt this run
            writer.writerow(['n1='+str(n1)+'; n2='+str(n2)+'. dx='+str(dx)+' over ['+str(left)+','+str(right)+']. N='+str(N)])

            # write data from all trials
            for i in range(len(resultsTable)):
                writer.writerow([resultsTable[i]])

    if timing:
        toc = time.perf_counter()
        print("runtime "+str(toc-tic))

    if showGraph:
        plt.hist(resultsTable, bins='auto')
        plt.show()


n1=1
n2=1
left=-20
right=20
dx=0.05
N=50
trials=25
makeOverlapsHisto(n1,n2,left,right,dx,N,trials)