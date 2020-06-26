import random
import numpy as np
import csv


def offdiagonal(mat):
    ''' mat should be a square, symmetric matrix
        returns the elements in the upper triangle (excluding the diagonal) '''
    dim = len(mat)
    offdiagList = [0] * int(dim*(dim-1)/2)
    n=0
    for i in range(dim):
        for j in range(i+1,dim):
            offdiagList[n] = mat[i][j]
            n+=1
    return offdiagList

random.seed()

D = 80 # dimension of the random matrix
Nlist = [10,100,500,1000,2500,5000] # options for number of random matrices to avg.
trials = 10 # numbers of SRIs to make for each N
resultsTable = []

for N in Nlist:
    # a row of results to record in my table
    # first entry in the row is N
    results = [N]

    # for each N, make, uh, 10 SRIs
    for j in range(trials):
        SRI = np.zeros((D,D),dtype=int)
        for i in range(N):
            zeta = [[random.choice([-1,1]) for i in range(D)]]
            outerprod = np.matmul(np.transpose(zeta), zeta)
            SRI = SRI + outerprod
        SRI = SRI*(1/N)
        offdiagList = offdiagonal(SRI)
        sigma = np.std(offdiagList)
        results.append(sigma)
    
    resultsTable.append(results)

with open('SRI_sds.csv','w',newline='') as csvFile:
    writer = csv.writer(csvFile, delimiter=',')
    for row in resultsTable:
        writer.writerow(row)