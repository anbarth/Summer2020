import random
import numpy as np
import csv
import math
import time
from linReg import regress

# imports that the UCSB computer doesnt support
import statistics
import matplotlib.pyplot as plt

    

def findIntercept(n1,n2,left,right,dx,Nlist,sampleSize,trials,writeToCsv=True,showGraph=True,timing=True):
    ### STEP 1: SET UP
    tic = time.perf_counter()

    # dimension of discretized position space
    D = int((right-left)/dx)

    # construct the phi and psi matrices
    psi = np.zeros((D,1))
    phi = np.zeros((1,D))

    # make hermite polynomial objects
    n1_arr = [0]*(n1+1)
    n1_arr[-1] = 1
    n2_arr = [0]*(n2+1)
    n2_arr[-1] = 1
    herm1 = np.polynomial.hermite.Hermite(n1_arr,[left,right])
    herm2 = np.polynomial.hermite.Hermite(n2_arr,[left,right])
    herm1_arr = herm1.linspace(n=D)[1]
    herm2_arr = herm2.linspace(n=D)[1]

    # psi and phi's norm-squareds, so i can normalize later
    norm1 = 0
    norm2 = 0

    x = left
    for i in range(D):
        psi[i][0] = math.exp(-1*x*x)*herm1_arr[i]
        phi[0][i] = math.exp(-1*x*x)*herm2_arr[i]
        norm1 += psi[i][0]*psi[i][0]
        norm2 += phi[0][i]*phi[0][i]
        x += dx
    # normalize
    psi = psi*(1/math.sqrt(norm1))
    phi = phi*(1/math.sqrt(norm2))



    ### STEP 2: RUN TRIALS
    sigResultsTable = []
    avgResultsTable = []
    for N in Nlist:
        # a row of results to record in the sigma table, in format:
        # 0 |  1   |      ...       | -4  |   -3    |   -2   |      -1
        # N | ln N | ... trials ... | avg | std err | ln avg | err in ln avg
        sigResults = [N]
        sigResults.append(np.log(N))

        # a row of results to record in the avgs table, in format:
        # 0 |  1   |      ...       | -2  |   -1      
        # N | ln N | ... trials ... | avg | std err 
        avgResults = [N]
        avgResults.append(np.log(N))

        # flesh out this row of the table by runing all trials
        for k in range(trials):
            # one trial: make ONE estimate of sigma, store in sigResults/one estimate of avg, store in avgResults

            overlaps = []
            for j in range(sampleSize):
                # one sample: make ONE estimate of the overlap, store in overlaps

                overlap = 0
                # calculate the overlap with N random vectors
                for i in range(N):
                    zeta = [[random.choice([-1,1])] for i in range(D)]
                    phizeta = np.matmul(phi, zeta) # <phi|z>
                    zetapsi = np.matmul(np.transpose(zeta), psi) # <z|psi>
                    prod = phizeta * zetapsi
                    overlap =  overlap + prod
                overlap =  overlap*(1/N) 
                # great! now store the result
                overlaps.append(overlap[0][0])
            
            # find average overlap, add to this row of the table
            avgResults.append(statistics.mean(overlaps)) 
            # find std dev of overlaps, add to this row of the table
            sigResults.append(statistics.stdev(overlaps))

        # finish off this row with some summary statistics
        sig_avg = statistics.mean(sigResults[2:])
        sig_ln_avg = np.log(sig_avg)
        sig_err = statistics.stdev(sigResults[2:]) / math.sqrt(trials)
        sig_ln_avg_err = sig_err/sig_avg
        sigResults.extend([sig_avg, sig_err, sig_ln_avg, sig_ln_avg_err])

        avg_avg = statistics.mean(avgResults[2:])
        avg_err = statistics.stdev(avgResults[2:]) / math.sqrt(trials)
        avgResults.extend([avg_avg, avg_err])

        # add this row to the table
        sigResultsTable.append(sigResults)
        avgResultsTable.append(avgResults)

    ### STEP 3: REGRESSION

    # here's the data i wanna work with
    lnN = [sigResultsTable[i][1] for i in range(len(sigResultsTable))]
    lnSigma = [sigResultsTable[i][-2] for i in range(len(sigResultsTable))]
    lnSigma_err = [sigResultsTable[i][-1] for i in range(len(sigResultsTable))]

    # regress!
    (slope, intercept, r_sq, slope_err, intercept_err) = regress(lnN, lnSigma)



    ### STEP 3: OUTPUT
    if writeToCsv:
        # sigmas
        # construct header row
        sigHeaderRow = ['N','ln N']
        for i in range(trials):
            sigHeaderRow.append(' ')
        sigHeaderRow.extend(['avg','std err','ln avg','err in ln avg'])

        with open('sigma_overlaps.csv','w',newline='') as csvFile:
            writer = csv.writer(csvFile, delimiter=',')

            # write specs abt this run
            writer.writerow(['n1='+str(n1)+'; n2='+str(n2)+'. dx='+str(dx)+' over ['+str(left)+','+str(right)+']'])
            writer.writerow(['sample size: '+str(sampleSize)+', '+str(trials)+' trials'])

            # write results of regression
            writer.writerow(['slope',str(slope),'slope err',str(slope_err)])
            writer.writerow(['int',str(intercept),'int err',str(intercept_err)])

            # write headers
            writer.writerow(sigHeaderRow)

            # write data from all trials
            for row in sigResultsTable:
                writer.writerow(row)

        # avgs
        # construct header row
        avgHeaderRow = ['N','ln N']
        for i in range(trials):
            avgHeaderRow.append(' ')
        avgHeaderRow.extend(['avg','std err'])

        with open('avg_overlaps.csv','w',newline='') as csvFile:
            writer = csv.writer(csvFile, delimiter=',')

            # write specs abt this run
            writer.writerow(['n1='+str(n1)+'; n2='+str(n2)+'. dx='+str(dx)+' over ['+str(left)+','+str(right)+']'])
            writer.writerow(['sample size: '+str(sampleSize)+', '+str(trials)+' trials'])

            # write headers
            writer.writerow(avgHeaderRow)

            # write data from all trials
            for row in avgResultsTable:
                writer.writerow(row)

    if timing:
        toc = time.perf_counter()
        print("runtime "+str(toc-tic))

    if showGraph:
        # titles and labels
        plt.xlabel('ln(N)')
        plt.ylabel('ln(sigma)')
        plt.title('n1='+str(n1)+'; n2='+str(n2)+'. dx='+str(dx)+' over ['+str(left)+','+str(right)+']')
        slope_str = str( int(slope*10000)/10000 ) + ' +/- ' + str( int(slope_err*10000)/10000 )
        int_str = str( int(intercept*1000)/1000 ) + ' +/- ' + str( int(intercept_err*1000)/1000 )
        r_sq_str = str( int(r_sq*1000)/1000 )
        plt.figtext(.6,.75,'slope: '+slope_str+'\nintercept: '+int_str+'\nR^2: '+r_sq_str)

        # plot data
        plt.errorbar(lnN,lnSigma,yerr=lnSigma_err,fmt='bo',capsize=4)

        # plot lin reg
        lnSig_model = [slope*x + intercept for x in lnN]
        plt.plot(lnN,lnSig_model)

        plt.show()

    return (intercept, intercept_err)

n1 = 1
n2 = 1
left = -10
right = 10
dx = 2
Nlist = [50,100,250]
sampleSize = 25
trials = 10

findIntercept(n1, n2, left, right, dx, Nlist, sampleSize, trials)

