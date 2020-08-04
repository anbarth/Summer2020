from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
#from sho import defectEigenstates, shoEigenbra, shoEigenket
import sho
import imp

imp.reload(sho)

left = -15
right = 15
dx = 0.05
# dimension of discretized position space
D = int((right-left)/dx)
domain = np.linspace(left,right,D,endpoint=False)

depth = 0
width = 0
center = 0

plotLeft = -6
plotRight = 6


nMax = 20

# define U
wing = width/2.0
U = np.zeros((D))
x = left
for i in range(D):
    pot = x*x
    if x <= center+wing and x >= center-wing:
        #pot = -1.0*depth
        pot -= depth
    U[i] = pot
    x += dx


(E,psi) = sho.defectEigenstates(depth,width,center,left,right,dx,0,nMax,0.01)


plt.clf()
plt.xlim(plotLeft,plotRight)
#plt.ylim(-95,-90)
#plt.ylim(-76,-71)
#plt.ylim(-44,-39)
#plt.ylim(-7,-2)
plt.ylim(0,20)
plt.plot(domain,U)
for n in range(nMax):
    psi_reg = sho.shoEigenket(n,dx,left,right)
    plt.plot(domain,[E[n]]*D,color='orange')
    if psi[n][int(D/6)] < 0:
        psi[n] = psi[n] * -1
    if psi_reg[int(D/6)] < 0:
        psi_reg = psi_reg * -1
    #psi[n] = psi[n]*3
    #psi_reg = psi_reg*3
    plt.plot(domain,3*psi[n]+E[n],color='black')
    #plt.plot(domain,3*psi_reg+E[n],color='green')


plt.show()


