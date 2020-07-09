import csv
import matplotlib.pyplot as plt
import numpy as np

fname = 'waluigi.csv'
nMax = 5
nLabel = ['1','','','','5']

title = ""
intercepts = []
with open(fname) as csvFile:
    reader = csv.reader(csvFile, delimiter='\t')
    line = 0
    for row in reader:
        if line == 0:
            title = row[0]
        else:
            interceptsRow = [float(x) for x in row]
            intercepts.append(interceptsRow)
        line += 1


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
plt.figtext(0.4,0.025,title)
plt.show()