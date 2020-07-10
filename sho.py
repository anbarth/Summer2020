import numpy as np
import math
def shoEigenket(n,dx,left,right):
    # dimension of discretized position space
    D = int((right-left)/dx)

    # construct the vector
    psi = np.zeros((D,1))

    # make hermite polynomial object
    n_arr = [0]*(n+1)
    n_arr[-1] = 1
    herm = np.polynomial.hermite.Hermite(n_arr,window=[left,right])
    herm_arr = herm.linspace(n=D)[1]

    # norm-squared, so i can normalize later
    norm2 = 0

    x = left
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
    herm = np.polynomial.hermite.Hermite(n_arr,window=[left,right])
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