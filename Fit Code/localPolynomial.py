from pyqt_fit.nonparam_regression import NonParamRegression
from pyqt_fit.npr_methods import LocalPolynomialKernel1D

from scipy.interpolate import UnivariateSpline

def localPoly(x, y, order, width):
	'''
	This method returns the local polynomial regression of order order using a running
	window of width width. The kernel used is a gaussian with bandwidth determine via
	the Scotts method.
	'''

	regressor = NonParamRegression(x, y, method=LocalPolynomialKernel1D(order))
	regressor.fit()

	predictions = regressor(x)
	params = (order, width)
	size = list(range(len(x)))
	predictor = regressor

	return predictions, params, size, predictor

def localLinear(x, y, width):
	return localPoly(x, y, 1, width)

def localQuadratic(x, y, width):
	return localPoly(x, y, 2, width)

def cubicSpline(x, y):
	predictor = UnivariateSpline(x, y, k=3)
	predictions = predictor(x)
	params = None
	size = list(range(len(x)))

	return predictions, params, size, predictor