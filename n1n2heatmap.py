import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import csv
from megaLogSigmaPlot import findIntercept
import time

random.seed()

nMax = 3
#nLabel = ['1','2','3']

left = -20
right = 20
dx = 0.5

Nlist = [50,150,500,1500,5000]
sampleSize = 500
trials = 10

intercepts = np.zeros((nMax,nMax))
intercept_errs = np.zeros((nMax,nMax))
slopes = np.zeros((nMax,nMax))
slope_errs = np.zeros((nMax,nMax))

tic = time.time()
totTime = 0

for n1 in range(1,nMax+1):
    for n2 in range(n1,nMax+1):
        (slope, slope_err, intercept, intercept_err) = findIntercept(n1, n2, left, right, dx, Nlist, sampleSize, trials,writeToCsv=False,showGraph=False)
        toc = time.time()
        print(str(toc-tic)+" s")
        totTime += toc-tic
        tic = toc
        #intercept = random.randint(-5,5)
        intercepts[n1-1][n2-1] = int(intercept*1000)/1000.0
        intercept_errs[n1-1][n2-1] = int(intercept_err*1000)/1000.0
        slopes[n1-1][n2-1] = int(slope*1000)/1000.0
        slope_errs[n1-1][n2-1] = int(slope_err*1000)/1000.0

print("total time: "+str(totTime)+ " s") 

'''fig, ax = plt.subplots()
im = ax.imshow(intercepts)

ax.set_xticks(np.arange(len(nLabel)))
ax.set_yticks(np.arange(len(nLabel)))
ax.set_xticklabels(nLabel)
ax.set_yticklabels(nLabel)

#plt.setp(ax.get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")

for i in range(nMax):
    for j in range(nMax):
        text = ax.text(j, i, intercepts[i][j], ha="center", va="center", color="w")
'''

with open('heatmap.csv','w') as csvFile:
#with open('heatmap.csv','w',newline='') as csvFile:
    writer = csv.writer(csvFile, delimiter=',')

    # write specs abt this run
    writer.writerow(['n1='+str(n1)+'; n2='+str(n2)+'. dx='+str(dx)+' over ['+str(left)+','+str(right)+']'])
    writer.writerow(['sample size: '+str(sampleSize)+', '+str(trials)+' trials'])

    # write intercepts
    writer.writerow(['intercepts'])
    for i in range(len(intercepts)):
        writer.writerow(intercepts[i])

    # write intercept errors
    writer.writerow(['intercept errors'])
    for i in range(len(intercept_errs)):
        writer.writerow(intercept_errs[i])

    # write slopes
    writer.writerow(['slopes'])
    for i in range(len(slopes)):
        writer.writerow(slopes[i])
    
    # write slope errors
    writer.writerow(['slope errors'])
    for i in range(len(slope_errs)):
        writer.writerow(slope_errs[i])


#fig.tight_layout()
#plt.show()
