from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
import imageio

left = -15
right = 15
dx = 0.05

# dimension of discretized position space
D = int((right-left)/dx)
domain = np.linspace(left,right,D,endpoint=False)

# retrict domain for plotting
# TODO assuming a symmetric interval.... for now
plotLeft = -5
plotRight = 5
mid = int(D/2)
wing = int(plotLeft/left/2 * D)
domain_r = domain[mid-wing:mid+wing]
D_r = len(domain_r)


def drawPlot(frameNum):
    plt.clf()
    plt.xlim(plotLeft,plotRight)
    plt.ylim(-4,10)
    # define U
    U = np.zeros((D))
    #wellBound = 0.25 + 2.5*np.abs(np.sin(0.05*frameNum))
    wellCenter = 2.5*np.sin(0.05*frameNum)
    x = left
    for i in range(D):
        pot = x*x
        #if x <= 1 and x >= -1:
        if x <= wellCenter+0.5 and x >= wellCenter-0.5:
            pot -= 3
        U[i] = pot
        x += dx

    # construct H
    ham = np.zeros((D,D))
    for i in range(D):
        # diagonal terms: U (potential)+2/dx^2 (kinetic)
        ham[i][i] = U[i]+2/(dx*dx)
        # tridiagonal terms: -1/dx^2 (kinetic)
        if i > 0:
            ham[i][i-1] = -1/(dx*dx)
            ham[i-1][i] = -1/(dx*dx)

    #E = eigh(ham,eigvals=(0,5),eigvals_only=True)
    (E,psi) = eigh(ham,eigvals=(0,7))
    psi = np.transpose(psi)

    #plt.plot(domain,U)
    U_r = U[mid-wing:mid+wing]
    plt.plot(domain_r,U_r)
    for i in range(7):
        plt.plot(domain_r,[E[i]]*D_r,color='orange')
        psi_r = psi[i][mid-wing:mid+wing]
        if psi_r[int(D_r/6)] < 0:
            psi_r = psi_r * -1
        #print(len(domain_r))
        #print(psi_r)
        plt.plot(domain_r,3*psi_r+E[i],color='black')

    figName = 'frames/'+str(frameNum)+'.png'
    plt.savefig(figName)
    #plt.show()
    return figName

#drawPlot(0)
images = []
for frameNum in range(126):
    figName = drawPlot(frameNum)
    images.append(imageio.imread(figName))
imageio.mimsave('gifs/movingWellWavefxns.gif', images)


