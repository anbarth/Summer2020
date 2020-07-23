import csv
import matplotlib.pyplot as plt
import numpy as np

# change these according to your needs
fname = 'heatmap.csv'
nMax = 10

title = ""
intercepts = []
intercept_errs = []
slopes = []
slope_errs = []

# read data out of the heatmap file
with open(fname) as csvFile:
    reader = csv.reader(csvFile, delimiter=',')
    line = 0
    for row in reader:
        if line == 0:
            title = row[0]
        elif line == 1:
            title += '\n'+row[0]
        # skip a line
        if line > 2 and line <= 2+nMax+1:
            thisRow = [float(x) for x in row]
            intercepts.append(thisRow)
        # skip a line
        if line > 3+nMax+1 and line <= 3+2*(nMax+1):
            thisRow = [float(x) for x in row]
            intercept_errs.append(thisRow)
        # skip a line
        if line > 4+2*(nMax+1) and line <= 4+3*(nMax+1):
            thisRow = [float(x) for x in row]
            slopes.append(thisRow)
        # skip a line
        if line > 5+3*(nMax+1) and line <= 5+4*(nMax+1):
            thisRow = [float(x) for x in row]
            slope_errs.append(thisRow)
        line += 1

slopeSigmaOff = np.zeros((nMax+1,nMax+1))
for n1 in range(nMax+1):
    for n2 in range(n1,nMax+1):
        slopeSigmaOff[n1][n2] = (slopes[n1][n2] + 0.5) / slope_errs[n1][n2]

logPercentError = np.zeros((nMax+1,nMax+1))
for n1 in range(nMax+1):
    for n2 in range(n1,nMax+1):
        logPercentError[n1][n2] = np.log(np.abs(intercept_errs[n1][n2] / intercepts[n1][n2]))

fig, ax = plt.subplots()
im = ax.imshow(intercepts)
title = "Intercepts\n"+title

nLabel = []
for n in range(nMax+1):
    if np.mod(n,5) == 0:
        nLabel.append(str(n))
    else:
        nLabel.append('')

ax.set_xticks(np.arange(len(nLabel)))
ax.set_yticks(np.arange(len(nLabel)))
ax.set_xticklabels(nLabel)
ax.set_yticklabels(nLabel)

plt.title(title)
plt.colorbar(im)

fig.tight_layout()
plt.show()
