import math
from myStats import mean, stdev
# output: slope, intercept, errors on each, R^2

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

    
    