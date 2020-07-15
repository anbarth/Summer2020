from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

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
        pot -= 3
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

(E,psi) = eigh(ham,eigvals=(0,8))
psi = np.transpose(psi)

# plot eigenfxn
domain = np.linspace(left,right,D,endpoint=False)

# retrict domain for plotting
# TODO assuming a symmetric interval.... for now
plotLeft = -4
plotRight = 4
mid = int(D/2)
wing = int(plotLeft/left/2 * D)
domain_r = domain[mid-wing:mid+wing]
U_r = U[mid-wing:mid+wing]
D_r = len(domain_r)

#plt.plot(domain,U)
plt.plot(domain_r,U_r)
for i in range(8):
    plt.plot(domain_r,[E[i]]*D_r,color='orange')


plt.show()


