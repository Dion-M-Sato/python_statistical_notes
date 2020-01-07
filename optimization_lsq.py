# https://python4mpia.github.io/fitting_data/least-squares-fitting.html#scipy-optimize-leastsq

import scipy.optimize as optimization
import numpy

# The function whose square is to be minimised.
# params ... list of parameters tuned to minimise function.
# Further arguments:
# xdata ... design matrix for a linear model.
# ydata ... observed data.
# Initial guess. x0
x0    = numpy.array([0.0, 0.0])
xdata = numpy.array([0.0,1.0,2.0,3.0,4.0,5.0])
ydata = numpy.array([0.1,0.9,2.2,2.8,3.9,5.1])

def func(params, xdata, ydata):
# Provide data as design matrix: straight line with a=0 and b=1 plus some noise.
    xdata = numpy.transpose(numpy.array([[1.0,1.0,1.0,1.0,1.0,1.0],[0.0,1.0,2.0,3.0,4.0,5.0]]))
    return(ydata - numpy.dot(xdata, params))

#This only provide fixed values of a and b.
print(optimization.leastsq(func, x0, args=(xdata, ydata)))
