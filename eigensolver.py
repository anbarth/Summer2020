from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from sho import shoEigenket

left = -15
right = 15
dx = 0.05

# dimension of discretized position space
D = int((right-left)/dx)

# define U
U = np.zeros((D))
x = left
for i in range(D):
    pot = x*x
    if x <= 1 and x >= -1:
        pot -= 20
    U[i] = pot
    x += dx

# construct H
ham = np.zeros((D,D))
for i in range(D):
    # diagonal terms: U (potential)+2/dx^2 (kinetic)
    ham[i][i] = U[i]+2/(dx*dx)
    # tridiagonal terms: -1/dx^2 (kinetic)
    if i > 0:
        ham[i][i-1] = -1/(dx*dx)
        ham[i-1][i] = -1/(dx*dx)

n=50

(E,psi) = eigh(ham,eigvals=(n,n+1))
psi = np.transpose(psi)

# plot eigenfxn
domain = np.linspace(left,right,D,endpoint=False)
psi_n = psi[0]
psiTrue = shoEigenket(n,dx,left,right) # analytic solution
plt.plot(domain,psi_n)
plt.plot(domain,psiTrue)
#plt.plot(domain,U)

#plt.legend(['numeric','analytic'])
plt.legend(['SHO w/ defect','SHO (analytic)'])
plt.title('n = '+str(n))
plt.show()

