from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
import sho
import importlib
importlib.reload(sho)

# this script displays an energy level diagram

############### INPUTS ##################

# bounds of position space
left = -15
right = 15

# mesh to put wavefunctions on
dx = 0.05
D = int((right-left)/dx) # dimension
domain = np.linspace(left,right,D,endpoint=False)

# mesh to solve on
dx_solve = 0.001
D_solve = int((right-left)/dx_solve) # dimension
domain_solve = np.linspace(left,right,D_solve,endpoint=False)

# define U
# see sho.py for lots of potentials to choose from
depth = 5
width = 3
center = 0
U = sho.defectPotential(depth,width,center,left,right,dx_solve)
#U = sho.flatDefectPotential(depth,width,center,left,right,dx_solve)

# eigenstates to solve for
nMax = 30
nMin = 0

# plot bounds
plotLeft = -4
plotRight = 4
plotBot = -6
plotTop = 9

##################################################

# get eigenstates
(E,psi) = sho.potentialEigenstates(U,nMin,nMax,left,right,dx,dx_solve)

# set up plot
plt.clf()
plt.xlim(plotLeft,plotRight)
plt.ylim(plotBot,plotTop)

# plot potential & wavefunctions
plt.plot(domain_solve,U,linewidth=2,color='gray')
for n in range(nMin,nMax+1):
    #plt.plot(domain,[E[n-nMin]]*D,color='orange')
    plt.plot(domain,5*psi[n-nMin]+E[n-nMin],color='black')


plt.xlabel('Position')
plt.ylabel('Potential energy')
plt.show()


