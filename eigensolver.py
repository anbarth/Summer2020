from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from sho import shoEigenket,defectEigenstates

# specify the discretized position space
left = -15
right = 15
dx = 0.05
# dimension of discretized position space
D = int((right-left)/dx)

# specify the defect to the SHO
depth = 10
width = 2
center = 0

# choose an energy level
n = 5

# ---------------------------------

# get SHO+defect eigenstate
(E,psi) = defectEigenstates(depth,width,center,left,right,dx,n,n)

# plot eigenfxns
domain = np.linspace(left,right,D,endpoint=False)

psi_n = psi[0] # SHO+defect solution
plt.plot(domain,psi_n)

psiTrue = shoEigenket(n,dx,left,right) # SHO analytic solution
plt.plot(domain,psiTrue)

plt.legend(['SHO w/ defect','SHO (analytic)'])
plt.title('n = '+str(n))
plt.show()