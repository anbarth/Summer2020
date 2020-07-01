import random
from logSigmaPlot import makeLogSigmaPlot


random.seed()

# choose two SHO energy levels
n1 = 1
n2 = 2

# bounds of discretized position space
left = -10
right = 10

# step size
dx = 2

# options for number of random matrices to avg.
Nlist = [10,50,250,1250,5000] 

# number of trials to take for each value of N
trials = 25

makeLogSigmaPlot(n1,n2,left,right,dx,Nlist,trials,timing=True,showGraph=True)