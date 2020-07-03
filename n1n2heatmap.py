import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from megaLogSigmaPlot import findIntercept

random.seed()

nMax = 5
nLabel = ['1','','','','5']

left = -10
right = 10
dx = 2

Nlist = [50,150,500]
sampleSize = 25
trials = 10

intercepts = np.zeros((nMax,nMax))

for n1 in range(1,nMax+1):
    for n2 in range(n1,nMax+1):
        #intercept = findIntercept(n1, n2, left, right, dx, Nlist, sampleSize, trials)
        intercept = random.randint(-5,5)
        intercepts[n1-1][n2-1] = intercept

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


fig.tight_layout()
plt.show()