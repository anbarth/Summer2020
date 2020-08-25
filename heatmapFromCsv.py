import csv
import matplotlib.pyplot as plt
import numpy as np

# this script reads a csv (the output from my heatmap maker script) and makes a heatmap


fname = 'data/aug 18/run10.csv' # read from this file
figname = 'data/aug 18/run10.png' # save to this file (if saveFig is True)
saveFig = False

# the max n in the file you're reading
# SET THIS MANUALLY!! otherwise, it'll crash
nMax = 20
# the max n to display on the intercepts heatmap
nMax_disp = nMax


title = ""
data = [[] for x in range(6)]

# read data out of the heatmap file
with open(fname) as csvFile:
    reader = csv.reader(csvFile, delimiter=',')
    line = 0
    for row in reader:
        # lines 0-1: run details
        if line == 0:
            title = row[0]+'\ndepth: '+row[1]+', width: '+row[2]+', center: '+row[3]
        #elif line == 1:
        #    title += '\n'+row[0]
        # lines 3-end: data
        for i in range(len(data)):
            if line > (i+2)+(i)*(nMax+1) and line <= (i+2)+(i+1)*(nMax+1):
                thisRow = [float(x) for x in row]
                data[i].append(thisRow)
                break
        line += 1


# mirror all data over diagonal
for i in range(len(data)):
    for n1 in range(nMax+1):
        for n2 in range(n1,nMax+1):
            data[i][n2][n1] = data[i][n1][n2]

# putting the data in named variables,
# just for the sake of readability
intercepts = data[0]
intercept_errs = data[1]
slopes = data[2]
slope_errs = data[3]
overlaps = data[4]
overlap_errs = data[5]

# slopes' number of sigma off from -0.5
slopeSigmaOff = np.zeros((nMax+1,nMax+1))
for n1 in range(nMax+1):
    for n2 in range(nMax+1):
        slopeSigmaOff[n1][n2] = (slopes[n1][n2] + 0.5) / slope_errs[n1][n2]

avgSlope = np.sum(slopes) / ((nMax+1)*(nMax+1))
print(avgSlope)

# overlaps' number of sigma off from 0 
# (all overlaps should be 0, because i subtract 1 along the diagonal)
overlapSigmaOff = np.zeros((nMax+1,nMax+1))
for n1 in range(nMax+1):
    for n2 in range(nMax+1):
        overlapSigmaOff[n1][n2] = overlaps[n1][n2] / overlap_errs[n1][n2]

# intercepts' log10(percent error), useful for getting an order of magnitude
logPercentError = np.zeros((nMax+1,nMax+1))
for n1 in range(nMax+1):
    for n2 in range(nMax+1):
        logPercentError[n1][n2] = np.log10(np.abs(intercept_errs[n1][n2] / intercepts[n1][n2]))

# "hide" the values along the diagonal
# because otherwise they outshine the rest of the heatmap
for i in range(len(intercepts)):
    # these are just some values that are typically around the middle of the range
    intercepts[i][i] = -0.02
    overlap_errs[i][i] = 0.0007

# add 1 back to the diagonal overlaps, to see the resolution of orthonormality in all its glory
for i in range(len(overlaps)):
    overlaps[i][i] = overlaps[i][i]+1

# restrict intercepts display to [0,nMax_disp]
intercepts_disp = np.zeros((nMax_disp+1,nMax_disp+1))
for i in range(len(intercepts_disp)):
    for j in range(len(intercepts_disp[i])):
        intercepts_disp[i][j] = intercepts[i][j]


fig, ax = plt.subplots()

# uncomment whatever you want to plot

im = ax.imshow(intercepts_disp)#,vmin=-0.12, vmax=-0.01)
title = "Intercepts\n"+title

#im = ax.imshow(intercept_errs)
#title = "Intercept errors\n"+title

#im = ax.imshow(logPercentError, vmin=-1, vmax=1)
#title = "Intercept log10(% error)\n"+title

#im = ax.imshow(slopes)
#title = "Slopes\n"+title

#im = ax.imshow(slope_errs)
#title = "Slope errors\n"+title

#im = ax.imshow(slopeSigmaOff)
#title = "Slope sigma off from -0.5\n"+title

#im = ax.imshow(overlaps,cmap='bone')
#title = "Overlaps\n"+title

#im = ax.imshow(overlap_errs)
#title = "Overlap errors\n"+title

#im = ax.imshow(overlapSigmaOff)
#title = "Overlap error in sigma\n"+title



nLabel = []
#for n in range(nMax+1):
for n in range(nMax_disp+1):
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

if saveFig:
    plt.savefig(figname)

plt.show()
