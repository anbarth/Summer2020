import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import csv
from megaLogSigmaPlot import findIntercept

random.seed()

nMax = 5
nLabel = ['1','','','','5']

left = -10
right = 10
dx = 4

Nlist = [50,150,500]
sampleSize = 25
trials = 10

intercepts = np.zeros((nMax,nMax))
intercept_errs = np.zeros((nMax,nMax))

for n1 in range(1,nMax+1):
    for n2 in range(n1,nMax+1):
        (intercept, intercept_err) = findIntercept(n1, n2, left, right, dx, Nlist, sampleSize, trials,writeToCsv=False,showGraph=False)
        #intercept = random.randint(-5,5)
        intercepts[n1-1][n2-1] = int(intercept*1000)/1000
        intercept_errs[n1-1][n2-1] = int(intercept_err*1000)/1000

fig, ax = plt.subplots()
im = ax.imshow(intercepts)

ax.set_xticks(np.arange(len(nLabel)))
ax.set_yticks(np.arange(len(nLabel)))
ax.set_xticklabels(nLabel)
ax.set_yticklabels(nLabel)

#plt.setp(ax.get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")

for i in range(nMax):
    for j in range(nMax):
        text = ax.text(j, i, intercepts[i][j], ha="center", va="center", color="w")

with open('heatmap.csv','w',newline='') as csvFile:
    writer = csv.writer(csvFile, delimiter=',')

    # write specs abt this run
    writer.writerow(['n1='+str(n1)+'; n2='+str(n2)+'. dx='+str(dx)+' over ['+str(left)+','+str(right)+']'])
    writer.writerow(['sample size: '+str(sampleSize)+', '+str(trials)+' trials'])

    # write heatmap data
    for i in range(len(intercepts)):
        dataRow = [str(int) for int in intercepts[i]] + [' '] + [str(err) for err in intercept_errs[i]]
        writer.writerow(dataRow)


fig.tight_layout()
plt.show()