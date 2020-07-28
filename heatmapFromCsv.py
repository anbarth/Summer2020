import csv
import matplotlib.pyplot as plt
import numpy as np

# change these according to your needs
fname = 'anthonyheatmap.csv'
nMax = 10

title = ""


data = [[] for x in range(8)]

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

        '''if line > 2 and line <= 2+nMax+1:
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
        # skip a line
        if line > 6+4*(nMax+1) and line <= 6+5*(nMax+1):
            thisRow = [float(x) for x in row]
            overlaps.append(thisRow)
        # skip a line
        if line > 7+5*(nMax+1) and line <= 7+6*(nMax+1):
            thisRow = [float(x) for x in row]
            overlap_errs.append(thisRow)'''
        


intercepts = data[0]
intercept_errs = data[1]
slopes = data[2]
slope_errs = data[3]
#overlaps = []
#overlap_errs = []
interceptsb = data[4]
interceptb_errs = data[5]
slopesb = data[6]
slopeb_errs = data[7]

slopeSigmaOff = np.zeros((nMax+1,nMax+1))
for n1 in range(nMax+1):
    for n2 in range(n1,nMax+1):
        slopeSigmaOff[n1][n2] = (slopes[n1][n2] + 0.5) / slope_errs[n1][n2]

logPercentError = np.zeros((nMax+1,nMax+1))
for n1 in range(nMax+1):
    for n2 in range(n1,nMax+1):
        logPercentError[n1][n2] = np.log(np.abs(intercept_errs[n1][n2] / intercepts[n1][n2]))

fig, ax = plt.subplots()
#im = ax.imshow(intercepts)
#title = "Intercepts\n"+title
#im = ax.imshow(interceptb_errs)
#title = "Intercept errors\n"+title
#im = ax.imshow(slopesb)
#title = "Slopes\n"+title
im = ax.imshow(slopeb_errs)
title = "Slope errors\n"+title
#im = ax.imshow(overlaps)
#im = ax.imshow(slopeSigmaOff)
#title = "Slope sigma off from -0.5\n"+title
#title = "Overlaps\n"+title
#im = ax.imshow(overlap_errs)
#title = "Overlap errors\n"+title


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
