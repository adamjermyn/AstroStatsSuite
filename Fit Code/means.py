import numpy as np
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d

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

def meanFit(x, y, M):
	xSorted = x[np.argsort(x)]
	ySorted = y[np.argsort(x)]
	indices = range(M // 2,len(x)- M // 2 + 1 - (M % 2))

	yM = RunningMean(ySorted, M)
	model = interp1d(xSorted[indices],yM,kind='linear',fill_value=np.nan,bounds_error=False)

	return yM, None, indices, model

def binnedMeanFit(x, y, binWidth):
	xSorted = x[np.argsort(x)]
	ySorted = y[np.argsort(x)]
	nBins = int((max(x)-min(x))/binWidth)
	yM, xB, _ = binned_statistic(xSorted, ySorted, statistic='mean',bins=nBins)
	xM = (xB[1:] + xB[:-1])/2

	model = interp1d(xM,yM,kind='linear',fill_value=np.nan,bounds_error=False)

	return yM, None, list(range(len(y))), model