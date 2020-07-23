from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from sho import defectEigenstates

left = -15
right = 15
dx = 0.05
# dimension of discretized position space
D = int((right-left)/dx)
domain = np.linspace(left,right,D,endpoint=False)

depth = 15
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
plt.ylim(-15,15)
plt.plot(domain,U)
for i in range(nMax):
    plt.plot(domain,[E[i]]*D,color='orange')
    if psi[i][int(D/6)] < 0:
        psi[i] = psi[i] * -1
    plt.plot(domain,3*psi[i]+E[i],color='black')


plt.show()


