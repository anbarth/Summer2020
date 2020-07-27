from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from sho import defectEigenstates, shoEigenbra, shoEigenket

left = -15
right = 15
dx = 0.05
# dimension of discretized position space
D = int((right-left)/dx)
domain = np.linspace(left,right,D,endpoint=False)

depth = 100
width = 1
center = 0

plotLeft = -5
plotRight = 5


nMax = 15

# define U
wing = width/2.0
U = np.zeros((D))
x = left
for i in range(D):
    pot = x*x
    if x <= center+wing and x >= center-wing:
        pot -= depth
    U[i] = pot
    x += dx


(E,psi) = defectEigenstates(depth,width,center,left,right,dx,0,nMax)


plt.clf()
plt.xlim(plotLeft,plotRight)
plt.ylim(-95,-90)
#plt.ylim(-76,-71)
#plt.ylim(-44,-39)
#plt.ylim(-7,-2)
#plt.ylim(0,100)
plt.plot(domain,U)
for n in range(nMax):
    psi_reg = shoEigenket(n,dx,left,right)
    plt.plot(domain,[E[n]]*D,color='orange')
    if psi[n][int(D/6)] < 0:
        psi[n] = psi[n] * -1
    if psi_reg[int(D/6)] < 0:
        psi_reg = psi_reg * -1
    #psi[n] = psi[n]*3
    #psi_reg = psi_reg*3
    plt.plot(domain,3*psi[n]+E[n],color='black')
    plt.plot(domain,3*psi_reg+E[n],color='green')


plt.show()


