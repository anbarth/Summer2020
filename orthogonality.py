import numpy as np
import math
import matplotlib.pyplot as plt
from sho import shoEigenbra,shoEigenket,defectEigenstates

def orthoCheck(nMax,dx,bound):
    depth = 0
    width = 0
    center = 0
    
    # dimension of discretized position space
    left = -bound
    right = bound
    D = int((right-left)/dx)

    (E,psi) = defectEigenstates(depth,width,center,left,right,dx,0,nMax)

    overlaps = np.zeros((nMax,nMax))

   

    for n1 in range(0,nMax):
        psi1 = psi[n1]

        for n2 in range(n1,nMax):
            psi2 = psi[n2]

            overlap = np.vdot(psi1,psi2)
            overlaps[n1][n2] = overlap

    ### make heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(overlaps)

    nLabel = []
    for n in range(nMax):
        if np.mod(n,5) == 0:
            nLabel.append(str(n))
        else:
            nLabel.append('')

    plt.colorbar(im)

    ax.set_xticks(np.arange(len(nLabel)))
    ax.set_yticks(np.arange(len(nLabel)))
    ax.set_xticklabels(nLabel)
    ax.set_yticklabels(nLabel)

    #plt.setp(ax.get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")
    #plt.figtext(0.4,0.025,'dx='+str(dx)+' over ['+str(left)+','+str(right)+']')
    plt.title('dx='+str(dx)+' over ['+str(left)+','+str(right)+']')

    fig.tight_layout()
    plt.show() 

orthoCheck(51,0.025,30)