import numpy as np
from scipy.stats import multivariate_normal
from astroML.density_estimation import XDGMM
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import matplotlib.pyplot as plt

# If errors aren't provided, we can estimate them from data.
def bandwidthEstimate(x, y):
	data = np.transpose(np.array([x,y]))

	# Cross Validation Maximum Likelihood used for bandwidth estimation
	k = KDEMultivariate(data,var_type='cc',bw='cv_ml')

	bandwidth = k.bw

	return bandwidth

# Perform extreme deconvolution
def compute_XD_results(x, y, dx, dy, n_components=6, n_iter=50):
	X = np.vstack([x,y]).T
	Xerr = np.zeros(X.shape + X.shape[-1:])
	diag = np.arange(X.shape[-1])
	Xerr[:, diag, diag] = np.vstack([dx ** 2, dy ** 2]).T
	clf = None
	while clf is None:
		try:
			clf = XDGMM(n_components, n_iter=n_iter,verbose=True)
			clf.fit(X, Xerr)
		except:
			print('Error: Singular Matrix. Retrying...')
			clf = None
	return clf

# Compute gridded distribution
def gridDist(clf, x, y, num=2000):
	'''
	clf objects have attributes mu and V, which give the mean and covariance
	of the component gaussians.
	'''
	mux = clf.mu[:,0]
	muy = clf.mu[:,1]

	minX = np.min(x)
	maxX = np.max(x)

	widthX = maxX - minX

	ranX = np.linspace(minX, maxX, num=num)

	minY = np.min(y)
	maxY = np.max(y)

	widthY = maxY - minY

	ranY = np.linspace(minY, maxY, num=num)

	X, Y = np.meshgrid(ranX, ranY, indexing='ij')

	grid = np.zeros(X.shape)

	weights = clf.alpha

	for i in range(len(weights)):
		rv = multivariate_normal(clf.mu[i], clf.V[i])
		grid += weights[i]*rv.pdf(np.dstack((X, Y)))

	return grid, minX, maxX, minY, maxY

def distance(grid1, grid2):
	return -np.log(np.sum(np.sqrt(grid1*grid2))/np.sqrt(np.sum(grid1)*np.sum(grid2)))

def findNumComponents(x, y, dx, dy, num=2000):
	n = 1
	clf1 = compute_XD_results(x, y, dx, dy, n_components=n)
	grid1, minX, maxX, minY, maxY = gridDist(clf1, x, y, num=num)

	dist = 1

	print('Entering component loop.')
	while dist > 0.1:
		clf0 = clf1
		grid0 = grid1
		clf1 = compute_XD_results(x, y, dx, dy, n_components=n+1)

		grid1,_,_,_,_ = gridDist(clf1, x, y, num=num)

		dist = distance(grid0, grid1)

		n += 1
		print(n, dist)

	return n - 1, clf0, grid0, minX, maxX, minY, maxY

def computeExp(clf, num=2000, grid=None):
	if grid is None:
		grid, minX, maxX, minY, maxY = gridDist(clf, num=num)
	else:
		grid, minX, maxX, minY, maxY = grid
	ranX = np.linspace(minX, maxX, num=num)
	ranY = np.linspace(minY, maxY, num=num)
	expX = np.sum(grid*ranY[np.newaxis,:],axis=1) / np.sum(grid,axis=1)
	expY = np.sum(grid*ranX[:,np.newaxis],axis=0) / np.sum(grid,axis=0)
	return expX, ranX, expY, ranY
