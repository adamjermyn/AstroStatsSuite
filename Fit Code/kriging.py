import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


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
