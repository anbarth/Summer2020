from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
import imageio
import sho
import importlib

importlib.reload(sho)

nMax = 10
left = -15
right = 15
dx = 0.05

# dimension of discretized position space
D = int((right-left)/dx)
domain = np.linspace(left,right,D,endpoint=False)

# retrict domain for plotting
plotLeft = -5
plotRight = 5

def drawPlot(frameNum):
    depth = 5
    width = 1
    center = np.sin(frameNum)
    wing = width/2.0
    
    plt.clf()
    plt.xlim(plotLeft,plotRight)
    plt.ylim(-4,10)

    # define U
    U = np.zeros((D))
    x = left
    for i in range(D):
        pot = x*x
        if x <= center+wing and x >= center-wing:
            pot -= depth
        U[i] = pot
        x += dx

    (E,psi) = sho.defectEigenstates(depth,width,center,left,right,dx,0,nMax,0.001)

    plt.plot(domain,U)
    for i in range(7):
        plt.plot(domain,[E[i]]*D,color='orange')
        if psi[i][int(D/6)] < 0:
            psi[i] = psi[i] * -1
        plt.plot(domain,3*psi[i]+E[i],color='black')

    figName = 'frames/'+str(frameNum)+'.png'
    plt.savefig(figName)
    #plt.show()
    return figName

#drawPlot(0)
images = []
for frameNum in range(628):
    figName = drawPlot(frameNum)
    images.append(imageio.imread(figName))
imageio.mimsave('gifs/movingWellWavefxns.gif', images)


