import numpy as np
import math
import matplotlib.pyplot as plt
from sho import shoEigenbra,shoEigenket

def orthoCheck(nMax,dx,bound):
    overlaps = np.zeros((nMax,nMax))

    # dimension of discretized position space
    left = -bound
    right = bound
    D = int((right-left)/dx)

    for n1 in range(0,nMax):
        # construct the psi vector...
        psi = shoEigenket(n1,dx,left,right)

        for n2 in range(n1,nMax):
            # construct the phi vector...
            phi = shoEigenbra(n2,dx,left,right)

            overlap = np.matmul(phi,psi)
            overlaps[n1][n2] = overlap[0][0]

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

orthoCheck(51,0.025,20)