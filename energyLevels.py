from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
#from sho import defectEigenstates, shoEigenbra, shoEigenket
import sho
import importlib

importlib.reload(sho)
left = -15
right = 15
dx = 0.05
# dimension of discretized position space
D = int((right-left)/dx)
domain = np.linspace(left,right,D,endpoint=False)

depth = 50
width = 2
center = 7

plotLeft = -15
plotRight = 15
plotBot = -10
plotTop = 40


nMax = 20
nMin = 0

# define U
wing = width/2.0
U = np.zeros((D))
x = left
for i in range(D):
    pot = x*x
    #pot = 0
    if x <= center+wing and x >= center-wing:
        #pot = -1.0*depth
        pot -= depth
    U[i] = pot
    x += dx


(E,psi) = sho.defectEigenstates(depth,width,center,left,right,dx,nMin,nMax,0.001)
#(E,psi) = sho.squareWellEigenstates(depth,left,right,dx,nMin,nMax,0.001)

plt.clf()
plt.xlim(plotLeft,plotRight)
plt.ylim(plotBot,plotTop)
#plt.ylim(-315,-300)

plt.plot(domain,U)
for n in range(nMin,nMax+1):
    psi_reg = sho.shoEigenket(n,dx,left,right)
    plt.plot(domain,[E[n-nMin]]*D,color='orange')
    if psi[n-nMin][int(D/6)] < 0:
        psi[n-nMin] = psi[n-nMin] * -1
    if psi_reg[int(D/6)] < 0:
        psi_reg = psi_reg * -1
    #psi[n-nMin] = psi[n-nMin]*3
    #psi_reg = psi_reg*3
    plt.plot(domain,3*psi[n-nMin]+E[n-nMin],color='black')
    #plt.plot(domain,3*psi_reg+E[n-nMin],color='green')


plt.show()


