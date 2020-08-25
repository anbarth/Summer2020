from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
import imageio
import sho
import importlib
importlib.reload(sho)

# this script makes an animated gif of a potential well & energy levels

numFrames = 10
nMax = 6

# bounds of position space
left = -15
right = 15

# right now, it's set up to show how the energy levels change with dx
# you could just set dx to a constant if you weren't trying to do that
dxs = np.linspace(0.3,0.001,num=numFrames)

# plot axes
plotLeft = -4
plotRight = 4
plotBot = -7
plotTop = 9

def drawPlot(frameNum):
    ''' creates a single frame of the gif '''

    dx = dxs[frameNum]

    D = int((right-left)/dx) # dimension
    domain = np.linspace(left,right,D,endpoint=False)

    # defect parameters
    # set to constants right now,
    # but you could make them depend on frameNum
    depth = 5
    width = 3
    center = 0
    wing = width/2.0

    # set up plot
    plt.clf()
    plt.xlim(plotLeft,plotRight)
    plt.ylim(plotBot,plotTop)

    # define U
    U = sho.defectPotential(depth,width,center,left,right,0.05)
    mydomain = np.linspace(plotLeft,plotRight,len(U),endpoint=False)
    plt.plot(mydomain,U)

    # progress bar
    progBar = [-6]*len(U)
    progLen = int( frameNum*1.0/numFrames * len(U) )
    plt.plot(mydomain[0:progLen],progBar[0:progLen],color='gray',linewidth=8)

    # eigenstates
    (E,psi) = sho.defectEigenstates(depth,width,center,left,right,0,nMax,dx,dx)

    for i in range(nMax):
        # plot energy level
        plt.plot(domain,[E[i]]*D,color='black')
        # this check prevents wavefunctions from flipping upside-down from frame to frame
        if psi[i][int(D/6)] < 0:
            psi[i] = psi[i] * -1   
        # plot wavefunction
        #plt.plot(domain,3*psi[i]+E[i],color='black')

    frameLabel = 'dx = '+str( int(dx*1000)/1000 )
    plt.figtext(0.15,0.175,frameLabel)

    figName = 'frames/'+str(frameNum)+'.png'
    plt.savefig(figName)
    return figName


images = []
# make all frames
for frameNum in range(numFrames):
    figName = drawPlot(frameNum)
    images.append(imageio.imread(figName))
# make the gif!
imageio.mimsave('gifs/waaa.gif', images)


