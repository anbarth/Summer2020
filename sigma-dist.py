import random
import numpy as np
import csv
from sampling import offdiagonal

# let's look at the actual distribution of sigmas

D = 40
N = 1000
trials = 1000
resultsTable = []

for j in range(trials):
    SRI = np.zeros((D,D),dtype=int)
    for i in range(N):
        zeta = [[random.choice([-1,1]) for i in range(D)]]
        outerprod = np.matmul(np.transpose(zeta), zeta)
        SRI = SRI + outerprod
    SRI = SRI*(1/N)
    offdiagList = offdiagonal(SRI)
    sigma = np.std(offdiagList)
    resultsTable.append(sigma)



with open('sigma_dist.csv','w',newline='') as csvFile:
    writer = csv.writer(csvFile, delimiter=',')
    for i in range(len(resultsTable)):
        writer.writerow([resultsTable[i]])