import numpy as np
import math
import matplotlib.pyplot as plt

def orthoCheck(nMax,dx,bound):
    overlaps = np.zeros((nMax,nMax))

    # dimension of discretized position space
    left = -bound
    right = bound
    D = int((right-left)/dx)

    for n1 in range(0,nMax):
        #if n1 != 0:
        #    continue
        # construct the psi vector...
        psi = np.zeros((D,1))
        # make hermite polynomial object
        n1_arr = [0]*(n1+1)
        n1_arr[-1] = 1
        herm1 = np.polynomial.hermite.Hermite(n1_arr,window=[left,right])
        herm1_arr = herm1.linspace(n=D)[1]
        # psi's norm-squared
        norm1 = 0
        x = left
        for i in range(D):
            psi[i][0] = math.exp(-1/2.0*x*x)*herm1_arr[i]
            norm1 += psi[i][0]*psi[i][0]
            x += dx
        # normalize
        psi = psi*(1/math.sqrt(norm1))

        for n2 in range(n1,nMax):
            #if n2 != 3:
            #    continue

            # construct the phi vector...
            phi = np.zeros((1,D))
            # make hermite polynomial object
            n2_arr = [0]*(n2+1)
            n2_arr[-1] = 1
            herm2 = np.polynomial.hermite.Hermite(n2_arr,window=[left,right])
            herm2_arr = herm2.linspace(n=D)[1]
            # phi's norm-squared
            norm2 = 0
            x = left
            for i in range(D):
                phi[0][i] = math.exp(-1/2.0*x*x)*herm2_arr[i]
                norm2 += phi[0][i]*phi[0][i]
                x += dx
            # normalize
            phi = phi*(1/math.sqrt(norm2))

            #print(np.matmul(phi,psi))
            overlap = np.matmul(phi,psi)
            '''if overlap[0][0] <= -0.5:
                print(str(n1)+" "+str(n2))'''
            overlaps[n1][n2] = overlap[0][0]

    #return
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