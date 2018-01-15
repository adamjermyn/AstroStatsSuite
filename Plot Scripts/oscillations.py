import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline

def krigingFit(x, y):
	'''
	We use a product of a constant and radial basis kernel.
	The constant kernel is used to allow a variable weight to be given to different points.
	The radial basis part makes sure the fit is local. 
	'''

	# Determine an appropriate bandwidth range for the radial basis kernel.
	dx = np.diff(sorted(list(x)))
	minD = min(dx)
	maxD = max(dx)
	r = RBF(np.mean(dx), (minD/100, maxD*100))

	# This puts a minimum weight on each point of 1e-3 and a maximum of 1e3.
	c = C(1.0, (1e-3, 1e3))

	kernel = c * r

	gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=np.sqrt(np.average(np.abs(y))))

	gp.fit(x[:,np.newaxis],y)

	def f(x):
		return gp.predict(x[:,np.newaxis])

	yPred = f(x)

	return yPred, gp, range(len(x)), f


def binnedMeanFit(x, y, binWidth):
	xSorted = x[np.argsort(x)]
	ySorted = y[np.argsort(x)]
	nBins = int((max(x)-min(x))/binWidth)
	yM, xB, _ = binned_statistic(xSorted, ySorted, statistic='mean',bins=nBins)
	xM = (xB[1:] + xB[:-1])/2

	model = interp1d(xM,yM,kind='linear',fill_value=np.nan,bounds_error=False)

	yM = model(x)

	return yM, None, list(range(len(y))), model

def cubicSpline(x, y):
	predictor = UnivariateSpline(x, y, k=3, s=100)
	predictions = predictor(x)
	params = None
	size = list(range(len(x)))

	return predictions, params, size, predictor

from sklearn.ensemble import RandomForestRegressor

def randomForest(x, y):
	regr = RandomForestRegressor(random_state=0)
	regr.fit(x[:,np.newaxis], y)

	def f(x):
		return regr.predict(x[:,np.newaxis])

	predictions = f(x)
	params = (regr,)
	size = list(range(len(x)))

	return predictions, params, size, f

def RunningMean(seq, M):
	"""
	 Purpose:
		Find the mean for the points in a sliding window (odd number in size) 
		as it is moved from left to right by one point at a time.
	  Inputs:
		seq -- 	array containing items for which a running mean (in a sliding window) 
			is to be calculated
		M -- number of items in window (window size) -- must be an integer > 1
	  Outputs:
		 medians -- list of means with size N - M + 1
	   Note:
		1. The median of a finite list of numbers is the "center" value when this list
		is sorted in ascending order. 
		2. If M is an even number the two elements in the window that
		are close to the center are averaged to give the median (this
		is not by definition)
	"""  
	# Repeat array M times along new dimension
	tiled = np.tile(seq,(M,1))
	for i in range(M):
		# Shift rows of repeated array by row number, so each column is the median window
		tiled[i] = np.roll(tiled[i],i)
	# We eliminate the first M-1 means because those have values that shifted 'around'
	# from the end of the array to the beginning.
	return np.mean(tiled,axis=0)[M-1:]

def RunningMedian(seq, M):
	"""
	 Purpose:
		Find the median for the points in a sliding window (odd number in size) 
		as it is moved from left to right by one point at a time.
	  Inputs:
		seq -- 	array containing items for which a running median (in a sliding window) 
			is to be calculated
		M -- number of items in window (window size) -- must be an integer > 1
	  Outputs:
		 medians -- list of medians with size N - M + 1
	   Note:
		1. The median of a finite list of numbers is the "center" value when this list
		is sorted in ascending order. 
		2. If M is an even number the two elements in the window that
		are close to the center are averaged to give the median (this
		is not by definition)
	"""  
	# Repeat array M times along new dimension
	tiled = np.tile(seq,(M,1))
	for i in range(M):
		# Shift rows of repeated array by row number, so each column is the median window
		tiled[i] = np.roll(tiled[i],i)
	# We eliminate the first M-1 medians because those have values that shifted 'around'
	# from the end of the array to the beginning.
	return np.median(tiled,axis=0)[M-1:]


def binnedMedianFit(x, y, nBins):
	xSorted = x[np.argsort(x)]
	ySorted = y[np.argsort(x)]

	yM, xB, _ = binned_statistic(xSorted, ySorted, statistic='median',bins=nBins)
	xM = (xB[1:] + xB[:-1])/2

	model = interp1d(xM,yM,kind='linear',fill_value=np.nan,bounds_error=False)

	predictions = model(xSorted)

	return predictions, None, list(range(len(y))), model

plt.subplot(211)

width = 4
nPer = 16000

x = np.linspace(-width/2,width/2,num=width*nPer,endpoint=True)

y0 = np.sin(2*np.pi*x)

plt.plot(x, y0, label='True Relationship')

y = y0 + 4 * np.random.randn(len(x))

plt.scatter(x,y, s=1, label='Data', c='k')

m1y = binnedMeanFit(x, y, 1.*(width + 1)/width)[0]

plt.plot(x, m1y, label='Binned Means')

m2y = binnedMeanFit(x[3*nPer//8:-nPer//8], y[3*nPer//8:-nPer//8], 1.*(width + 1)/width)[0]

plt.plot(x[3*nPer//8:-nPer//8], m2y, label='Offset Binned Means')

p = binnedMeanFit(x, y, 1./20)[0]

plt.plot(x, p, label='Narrow Binned Means')


plt.subplot(212)

plt.plot(x, y0, label='True Relationship')
plt.plot(x, m1y, label='Binned Medians')
plt.plot(x[nPer//4:-nPer//4], m2y, label='Offset Binned Means')
plt.plot(x, p, label='Narrow Binned Means')
plt.ylim([-1,1])

plt.legend()
plt.tight_layout()

plt.savefig('../Plots/oscillations.pdf')

plt.show()
