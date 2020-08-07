import numpy as np
import math
from scipy.linalg import eigh,eigh_tridiagonal

def cakeEigenstates(d1,d2,w1,w2,center,left,right,dx,nMin,nMax,dx_solve):
    # dimension of discretized position space in which we SOLVE
    D_solve = int((right-left)/dx_solve)
    wing1 = w1*1/2.0
    wing2 = w2*1/2.0


    # define U
    U = np.zeros((D_solve))
    x = left
    for i in range(D_solve):
        pot = 0
        if x >= -1*wing1 and x <= wing1:
            pot -= d1
            if x >= center-wing2 and x <= center+wing2:
                pot -= d2
        U[i] = pot
        x += dx_solve

    # construct H
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

def flatDefectEigenstates(depth,width,center,left,right,dx,nMin,nMax,dx_solve):
    # dimension of discretized position space in which we SOLVE
    D_solve = int((right-left)/dx_solve)
    wing = width/2.0

    # define U
    U = np.zeros((D_solve))
    x = left
    for i in range(D_solve):
        pot = x*x
        if x <= center+wing and x >= center-wing:
            pot = -1.0*depth
        U[i] = pot
        x += dx_solve

    # construct H
    diag = U + 2/(dx_solve*dx_solve)
    tridiag = [-1.0/(dx_solve*dx_solve)] * (D_solve-1)

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
