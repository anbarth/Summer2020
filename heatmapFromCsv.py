import csv
import matplotlib.pyplot as plt
import numpy as np

fname = 'heatmap.csv'
nMax = 5

title = ""
intercepts = []
with open(fname) as csvFile:
    reader = csv.reader(csvFile, delimiter=',')
    line = 0
    for row in reader:
        if line > 2 and line <= 3+nMax:
            interceptsRow = [float(x) for x in row]
            intercepts.append(interceptsRow)
        elif line == 0:
            title = row[0]
        elif line == 1:
            title += '\n'+row[0]
        # do nothing with line 2  
        line += 1


fig, ax = plt.subplots()
im = ax.imshow(intercepts)

nLabel = []
for n in range(nMax):
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