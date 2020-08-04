import csv
import matplotlib.pyplot as plt
import numpy as np

# change these according to your needs
#fname = 'data/july 30/arrayAvg.csv'
#fname = 'data/july 31/731run3.csv'
fname = 'data/aug 3/83run3.csv'
#fname = 'theheatmap.csv'
nMax = 20

title = ""


data = [[] for x in range(6)]

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
        for i in range(len(data)):
            if line > (i+2)+(i)*(nMax+1) and line <= (i+2)+(i+1)*(nMax+1):
                thisRow = [float(x) for x in row]
                data[i].append(thisRow)
                break
        line += 1

        


intercepts = data[0]

for n1 in range(nMax+1):
    for n2 in range(n1,nMax+1):
        intercepts[n2][n1] = intercepts[n1][n2]

intercept_errs = data[1]
slopes = data[2]
slope_errs = data[3]
overlaps = data[4]
overlap_errs = data[5]


slopeSigmaOff = np.zeros((nMax+1,nMax+1))
for n1 in range(nMax+1):
    for n2 in range(n1,nMax+1):
        slopeSigmaOff[n1][n2] = (slopes[n1][n2] + 0.5) / slope_errs[n1][n2]

avgSlope = np.sum(slopes) / ((nMax+1)*(nMax+2)/2.0)
print(avgSlope)

logPercentError = np.zeros((nMax+1,nMax+1))
for n1 in range(nMax+1):
    for n2 in range(n1,nMax+1):
        logPercentError[n1][n2] = np.log10(np.abs(intercept_errs[n1][n2] / intercepts[n1][n2]))

hideDiagonal = False
if(hideDiagonal):
    for i in range(len(intercepts)):
        intercepts[i][i] = 0

# add 1 back to the diagonal overlaps
for i in range(len(overlaps)):
    overlaps[i][i] = overlaps[i][i]+1


fig, ax = plt.subplots()
#im = ax.imshow(intercepts,vmin=-0.07, vmax=0.03)#,)#cmap="jet",
#title = "Intercepts\n"+title
#im = ax.imshow(intercept_errs)
#title = "Intercept errors\n"+title
#im = ax.imshow(slopes)
#title = "Slopes\n"+title
#im = ax.imshow(slope_errs)
#title = "Slope errors\n"+title
im = ax.imshow(overlaps)
title = "Overlaps\n"+title
#im = ax.imshow(slopeSigmaOff)
#title = "Slope sigma off from -0.5\n"+title
#im = ax.imshow(overlap_errs)
#title = "Overlap errors\n"+title
#im = ax.imshow(logPercentError)#, vmin=-1.5, vmax=0.5)
#title = "Intercept log10(% error)\n"+title

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
