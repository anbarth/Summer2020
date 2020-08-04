import numpy as np
import math
from scipy.linalg import eigh,eigh_tridiagonal

def defectEigenstates(depth,width,center,left,right,dx,nMin,nMax,dx_solve):
    # dimension of discretized position space in which we SOLVE
    D_solve = int((right-left)/dx_solve)
    wing = width/2.0

    # define U
    U = np.zeros((D_solve))
    x = left
    for i in range(D_solve):
        pot = x*x
        if x <= center+wing and x >= center-wing:
            #pot = -1.0*depth
            pot -= depth
        U[i] = pot
        x += dx_solve

    # construct H
    '''ham = np.zeros((D_solve,D_solve))
    for i in range(D_solve):
        # diagonal terms: U (potential)+2/dx^2 (kinetic)
        ham[i][i] = U[i]+2/(dx_solve*dx_solve)
        # tridiagonal terms: -1/dx^2 (kinetic)
        if i > 0:
            ham[i][i-1] = -1/(dx_solve*dx_solve)
            ham[i-1][i] = -1/(dx_solve*dx_solve)'''
    diag = U + 2/(dx_solve*dx_solve)
    tridiag = [-1.0/(dx_solve*dx_solve)] * (D_solve-1)

    #(E,psi_smooth) = eigh(ham,eigvals=(nMin,nMax+1))
    (E,psi_smooth) = eigh_tridiagonal(diag,tridiag,select='i',select_range=(nMin,nMax))
    psi_smooth = np.transpose(psi_smooth)

    # now that it's been solved in the less discretized space,
    # transfer the wavefxns into the more discretized space
    D = int((right-left)/dx)
    psi = np.zeros((nMax-nMin+1,D))
    for i in range(D):
        pos = int(i*dx * 1.0/dx_solve)
        for j in range(len(psi)):
            psi[j][i] = psi_smooth[j][pos]

    # finally, normalize each wavefxn
    for i in range(len(psi)):
        norm2 = 0
        for j in range(D):
            norm2 += psi[i][j]*psi[i][j]
        # normalize
        psi[i] = psi[i]*(1/math.sqrt(norm2))

    return (E,psi)

def shoEigenket(n,dx,left,right):
    # dimension of discretized position space
    D = int((right-left)/dx)

    # construct the vector
    psi = np.zeros((D,1))

    # make hermite polynomial object
    n_arr = [0]*(n+1)
    n_arr[-1] = 1
    herm = np.polynomial.hermite.Hermite(n_arr,window=[left,right-dx])
    herm_arr = herm.linspace(n=D)[1]

    # norm-squared, so i can normalize later
    norm2 = 0

    x=left
    for i in range(D):
        psi[i][0] = math.exp(-1/2.0*x*x)*herm_arr[i]
        norm2 += psi[i][0]*psi[i][0]
        x += dx
    # normalize
    psi = psi*(1/math.sqrt(norm2))
    return psi

def shoEigenbra(n,dx,left,right):
    # dimension of discretized position space
    D = int((right-left)/dx)

    # construct the vector
    phi = np.zeros((1,D))

    # make hermite polynomial object
    n_arr = [0]*(n+1)
    n_arr[-1] = 1
    herm = np.polynomial.hermite.Hermite(n_arr,window=[left,right-dx])
    herm_arr = herm.linspace(n=D)[1]

    # norm-squared, so i can normalize later
    norm2 = 0

    x = left
    for i in range(D):
        phi[0][i] = math.exp(-1/2.0*x*x)*herm_arr[i]
        norm2 += phi[0][i]*phi[0][i]
        x += dx
    # normalize
    phi = phi*(1/math.sqrt(norm2))
    return phi
