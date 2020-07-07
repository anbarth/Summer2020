import math

def mean(L):
    tot = 0
    for x in L:
        tot += x
    return tot/len(L)

def stdev(L):
    avg = mean(L)
    sumOfSq = 0
    for x in L:
        sumOfSq += (x-avg)*(x-avg)
    return math.sqrt(sumOfSq / (len(L)-1))