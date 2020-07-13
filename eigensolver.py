from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from sho import shoEigenket

left = -10
right = 10
dx = 0.05

# dimension of discretized position space
D = int((right-left)/dx)

# define U
U = np.zeros((D))
x = left
for i in range(D):
    U[i] = x*x
    x += dx

# just a demo of eigh
'''diag = [1,1,1,1]
subd = [1,1,1]
mat = np.zeros((D,D))
for i in range(D):
    mat[i][i] = diag[i]
    if i > 0:
        mat[i][i-1] = subd[i-1]
        mat[i-1][i] = subd[i-1]

(w,v) = eigh(mat,eigvals=(0,1))
v = np.transpose(v)
print(w[1]*v[1])
print(np.matmul(mat,v[1]))'''

# construct H
ham = np.zeros((D,D))
for i in range(D):
    # diagonal terms: U (potential)+2/dx^2 (kinetic)
    ham[i][i] = U[i]+2/(dx*dx)
    # tridiagonal terms: -1/dx^2 (kinetic)
    if i > 0:
        ham[i][i-1] = -1/(dx*dx)
        ham[i-1][i] = -1/(dx*dx)

(E,psi) = eigh(ham,eigvals=(50,51))
psi = np.transpose(psi)

# plot eigenfxn
n = 50

domain = np.linspace(left,right,D,endpoint=False)
psi_n = psi[0]
#psiTrue = shoEigenket(n,dx,left,right) # analytic solution
plt.plot(domain,psi_n)
#plt.plot(domain,psiTrue)
plt.legend(['numeric','analytic'])
plt.title('n = '+str(n))
plt.show()

