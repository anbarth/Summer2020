import numpy as np
import math
from scipy.linalg import eigh,eigh_tridiagonal

# this file defines potentials & solves for energy eigenstates!


####### POTENTIALS ############

def nestedFiniteWellPotential(d1,d2,w1,w2,center,left,right,dx_solve):
    ''' returns a nested finite square well potential
        d1: depth of the upper well
        d2: depth of the lower well
        w1: width of the upper well
        w2: width of the lower well 
        center: center position of the lower well
        [left, right]: region over which to define the potential
        dx_solve: mesh size '''

    # dimension of discretized position space
    D_solve = int((right-left)/dx_solve)
    wing1 = w1*1/2.0
    wing2 = w2*1/2.0

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
    return U

def finiteSquareWellPotential(depth,width,center,left,right,dx_solve):
    ''' returns a finite square well nested in an infinite square wel potential
        depth: depth of the finite well
        width: width of the finite well
        center: center of the finite well
        [left, right]: region over which to define the potential; also, bounds of the infinite well
        dx_solve: mesh size '''

    # dimension of discretized position space
    D_solve = int((right-left)/dx_solve)
    wing1 = w1*1/2.0
    wing2 = w2*1/2.0

    # define U
    U = np.zeros((D_solve))
    x = left
    for i in range(D_solve):
        pot = 0
        if x >= -1*wing and x <= wing:
            pot -= depth
        U[i] = pot
        x += dx_solve
    
    return U

def flatDefectPotential(depth,width,center,left,right,dx_solve):
    ''' returns a SHO+defect potential, but the defect is flat on the bottom instead of curved
        depth: y-coordinate of the bottom of the defect
        width: width of defect
        center: center position of defect
        [left, right]: region where potential is defined
        dx_solve: mesh size '''

    # dimension of discretized position space
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
    
    return U

def defectPotential(depth,width,center,left,right,dx_solve):
    ''' returns a SHO+defect potential
        depth: depth of defect
        width: width of defect
        center: center position of defect
        [left, right]: region where potential is defined
        dx_solve: mesh size '''

    # dimension of discretized position space
    D_solve = int((right-left)/dx_solve)
    wing = width/2.0

    # define U
    U = np.zeros((D_solve))
    x = left
    for i in range(D_solve):
        pot = x*x
        if x <= center+wing and x >= center-wing:
            pot -= depth
        U[i] = pot
        x += dx_solve
    
    return U


####### EIGENSOLVERS ###########

def potentialEigenstates(U,nMin,nMax,left,right,dx,dx_solve):
    ''' given a potential U, returns the energy levels and eigenstates
        gives energy levels numbered nMin (inclusive) through nMax (inclusive)
        solves schrodinger's equation with mesh size dx_solve
        returns wavefunctions with mesh size dx '''

    # construct H from given U
    diag = U + 2/(dx_solve*dx_solve)
    tridiag = [-1.0/(dx_solve*dx_solve)] * (len(U)-1)

    # solve schrodinger's equation in the less discretized space
    (E,psi_smooth) = eigh_tridiagonal(diag,tridiag,select='i',select_range=(nMin,nMax))
    psi_smooth = np.transpose(psi_smooth)

    # transfer the wavefxns into the more discretized space
    D = int((right-left)/dx) # dimension of space
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

def nestedFiniteWellEigenstates(d1,d2,w1,w2,center,left,right,nMin,nMax,dx,dx_solve):
    ''' returns energy levels and eigenstates of a nested finite square well potential
        d1: depth of the upper well
        d2: depth of the lower well
        w1: width of the upper well
        w2: width of the lower well 
        center: center position of the lower well
        [left, right]: region over which to define the potential
        dx_solve: mesh size to use when solving schrodinger's eqn
        dx: mesh size to use for returned wavefunctions '''
    U = nestedFiniteWellPotential(d1,d2,w1,w2,center,left,right,dx_solve)
    return potentialEigenstates(U,nMin,nMax,left,right,dx,dx_solve)
    
def finiteSquareWellEigenstates(depth,width,center,left,right,nMin,nMax,dx,dx_solve):
    ''' returns energy levels and eigenstates of a nested finite square well potential
        depth: depth of the finite well
        width: width of the finite well
        center: center position of the finite well
        [left, right]: region over which to define the potential; also bounds of the infinite well
        dx_solve: mesh size to use when solving schrodinger's eqn
        dx: mesh size to use for returned wavefunctions '''

    U = finiteSquareWellPotential(depth,width,center,left,right,dx_solve)
    return potentialEigenstates(U,nMin,nMax,left,right,dx,dx_solve)
    
def defectEigenstates(depth,width,center,left,right,nMin,nMax,dx,dx_solve):
    ''' returns energy levels and eigenstates of a SHO+defect potential
        depth: depth of the defect
        width: width of the defect
        center: center position of the defect
        [left, right]: region where the potential is defined
        dx_solve: mesh size to use when solving schrodinger's eqn
        dx: mesh size to use for returned wavefunctions '''

    U = defectPotential(depth,width,center,left,right,dx_solve)
    return potentialEigenstates(U,nMin,nMax,left,right,dx,dx_solve)

def shoEigenket(n,dx,left,right):
    ''' returns an ANALYTIC solution psi to the SHO potential
        n: energy level
        dx: mesh size
        [left,right]: region where psi is defined 
    '''

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
