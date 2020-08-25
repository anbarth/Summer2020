import math

# anna's homemade stats package

# gives the mean of an iterable
def mean(L):
    tot = 0
    for x in L:
        tot += x
    return tot/len(L)

# gives the standard deviation of an iterable
# _sample_ standard deviation, not population!
def stdev(L):
    if len(L) < 2:
        return -1
    avg = mean(L)
    sumOfSq = 0
    
    for x in L:
        sumOfSq += (x-avg)*(x-avg)
    return math.sqrt(sumOfSq / (len(L)-1))

# does a linear regression on x_arr (independent) and y_arr (dependent)
# output: slope, intercept, R^2, error on slope, error on intercept
# NOTE i am somewhat suspicious of the formula for intercept uncertainty used here (see july 20 log)
def regress(x_arr, y_arr):
    
    y_bar = mean(y_arr)
    x_bar = mean(x_arr)
    
    SSxx = 0
    SSxy = 0
    SST = 0
    for i in range(len(x_arr)):
        x = x_arr[i]
        y = y_arr[i]
        SSxx += (x-x_bar)*(x-x_bar)
        SSxy += (x-x_bar)*(y-y_bar)
        SST += (y-y_bar)*(y-y_bar)
    
    m = SSxy/SSxx
    b = y_bar - m*x_bar
    y_hat_arr = [m*x+b for x in x_arr]

    SSE = 0
    for i in range(len(y_arr)):
        y = y_arr[i]
        y_hat = y_hat_arr[i]
        SSE += (y-y_hat)*(y-y_hat)

    R2 = (SST-SSE)/SST
    syx2 = SSE / (len(x_arr)-2)
    sm = math.sqrt( syx2 / SSxx )
    sb = math.sqrt( syx2 * (1.0/len(x_arr) + x_bar*x_bar / SSxx) )

    return (m, b, R2, sm, sb)

# does a linear regression on x_arr (independent) and y_arr (dependent), fixing the slope value
# returns the slope (as given), the intercept, and the R^2
def regressFixedSlope(x_arr,y_arr,slope):

    y_bar = mean(y_arr)
    x_bar = mean(x_arr)

    if(slope != False):
        b = y_bar - slope*x_bar
        y_hat_arr = [slope*x+b for x in x_arr]
        SST = 0
        SSE = 0
        for i in range(len(x_arr)):
            y = y_arr[i]
            y_hat = y_hat_arr[i]
            SST += (y-y_bar)*(y-y_bar)
            SSE += (y-y_hat)*(y-y_hat)
        R2 = (SST-SSE)/SST

        return (slope, b, R2)