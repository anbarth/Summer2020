import importlib
import sho
import numpy as np
import matplotlib.pyplot as plt
import myStats
importlib.reload(myStats)
importlib.reload(sho)

# this script tests for when energy levels converge, across different mesh sizes

# potential parameters
depth = 100
width = 1
center = 0
left = -20
right = 20

# energy levels to include
nMin = 0
nMax = 10

# range of mesh sizes to cover
dxs = np.linspace(0.1,0.01,num=100)

# desired percent convergence (for more detailed explanation of convergence criterion, see 8/14 work log)
convergenceLevel = 0.01
# size of moving regression window, in number of points
runningWindow = 30 


eigenvals = np.zeros((nMax+1-nMin, len(dxs))) # store all the energy levels at each mesh size
nConverged = np.zeros((nMax+1-nMin)) # keep track of which energy levels have converged already

# go through all the mesh sizes
for i in range(len(dxs)):
    dx = dxs[i]
    D = int((right-left)/dx) # dimension

    (E,psi) = sho.defectEigenstates(5,3,0,left,right,nMin,nMax,dx,dx)

    # store the energies
    for n in range(nMax+1-nMin):
        eigenvals[n][i] = E[n]

    # can't regress & decide you've converged if there havent been enough points yet
    if i < runningWindow-1:
        continue
    
    # now look at all the energy levels that haven't converged yet
    for n in range(nMax+1-nMin):
        if nConverged[n] == 0:
            # regress over all points within the regression windows
            (slope, b, R2, sm, sb) = myStats.regress(dxs[i-runningWindow+1 : i+1], eigenvals[n][i-runningWindow+1 : i+1])
            E0 = eigenvals[n][i] + slope*dxs[i] # estimate E(dx=0) from the regression model

            ratio = E0/eigenvals[n][i]
            if ratio < 1+convergenceLevel and ratio > 1-convergenceLevel:
                print('n='+str(n+nMin)+' converged by dx='+str(dx))
                nConverged[n] = dx


# PLOT RESULTS
plt.clf()

# plot the energy levels over the different mesh sizes
for n in range(nMax+1-nMin):
    plt.plot(dxs,eigenvals[n])

plt.xlim(max(dxs),min(dxs))
plt.xlabel('mesh size')
plt.ylabel('energy eigenval')

# plot each energy level's convergence dx
plt.figure()
nList = np.arange(nMin,nMax+1,1)
plt.plot(nList,nConverged)
plt.xlabel('n')
plt.ylabel('mesh size')

plt.show()