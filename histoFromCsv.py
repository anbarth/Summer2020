import csv
import matplotlib.pyplot as plt
import numpy as np

# this script reads a csv (the output from my heatmap maker script) and makes a histogram of intercepts


fname = 'data/aug 17/run1.csv' # read from this file
figname = 'data/aug 17/histo-run1' # save to this file (if saveFig is True)
saveFig = True

# the max n in the file you're reading
nMax = 20
# the max n to include in the histogram
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
            title = row[0]#+'\ndepth: '+row[1]+', width: '+row[2]+', center: '+row[3]
        elif line == 1:
            title += '\n'+row[0]
        # lines 3-end: data
        for i in range(len(data)):
            if line > (i+2)+(i)*(nMax+1) and line <= (i+2)+(i+1)*(nMax+1):
                thisRow = [float(x) for x in row]
                data[i].append(thisRow)
                break
        line += 1

# mirror over diagonal
for i in range(len(data)):
    for n1 in range(nMax+1):
        for n2 in range(n1,nMax+1):
            data[i][n2][n1] = data[i][n1][n2]

intercepts = data[0]

# "hide" the values along the diagonal
hideDiagonal = True
if(hideDiagonal):
    for i in range(len(intercepts)):
        intercepts[i][i] = 0

# restrict intercepts incldued to [0,nMax_disp]
intercepts_disp = np.zeros((nMax_disp+1,nMax_disp+1))
for i in range(len(intercepts_disp)):
    for j in range(len(intercepts_disp[i])):
        intercepts_disp[i][j] = intercepts[i][j]

# pool together all the intercepts
allIntercepts = []
for i in range(len(intercepts_disp)):
    allIntercepts.extend(intercepts_disp[i])

# make the histogram
plt.clf()
plt.hist(allIntercepts, bins='auto')
plt.title(title)
plt.xlabel('intercept')
plt.xlim(-0.07,0.02)

if saveFig:
    plt.savefig(figname)
plt.show()