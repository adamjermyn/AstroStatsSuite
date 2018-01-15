import numpy as np
from scipy.interpolate import interp1d

def ctdFit(x, y):
	ySorted = y[np.argsort(x)]
	xSorted = x[np.argsort(x)]
	indices = range(len(x))
	model = interp1d(xSorted, ySorted,kind='linear',fill_value=np.nan,bounds_error=False)

	return ySorted, None, indices, model
