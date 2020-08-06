from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
import sho
import importlib

importlib.reload(sho)
left = -15
right = 15
dx = 0.05
# dimension of discretized position space
D = int((right-left)/dx)
domain = np.linspace(left,right,D,endpoint=False)

d1 = 100
d2 = 70
w1 = 10
w2 = 0.5
center = 0

plotLeft = -15
plotRight = 15
plotBot = -201
plotTop = -10


nMax = 20
nMin = 0

wing1 = w1*1/2.0
wing2 = w2*1/2.0

# define U
U = np.zeros((D))
x = left
for i in range(D):
    pot = 0
    if x >= -1*wing1 and x <= wing1:
        pot -= d1
        if x >= center-wing2 and x <= center+wing2:
            pot -= d2
    U[i] = pot
    x += dx


(E,psi) = sho.cakeEigenstates(d1,d2,w1,w2,center,left,right,dx,nMin,nMax,0.001)

plt.clf()
plt.xlim(plotLeft,plotRight)
plt.ylim(plotBot,plotTop)

plt.plot(domain,U)
for n in range(nMin,nMax+1):
    plt.plot(domain,[E[n-nMin]]*D,color='orange')
    if psi[n-nMin][int(D/6)] < 0:
        psi[n-nMin] = psi[n-nMin] * -1
    #psi[n-nMin] = psi[n-nMin]*3
    plt.plot(domain,3*psi[n-nMin]+E[n-nMin],color='black')


plt.show()


