import numpy as np
from sklearn.linear_model import LassoLarsCV, LassoLarsIC, LassoCV, Lasso

def lassoFit(x, y):
	# Sort
	ind = np.argsort(x)
	xx = x[ind]
	yy = y[ind]

	x = xx
	y = yy

	# Construct feature matrix
	xMatrix = np.subtract.outer(x,x)
	xMatrix[xMatrix <= 0] = 0
	xMatrix = np.hstack((xMatrix,xMatrix**2))

	# Fit
	model = LassoCV(max_iter=100000, verbose=True, n_jobs=-1, normalize=True, fit_intercept=True)
	model.fit(xMatrix,y)

	print(model.alpha_, np.sum(np.abs(model.coef_)>0),model.coef_)

	# Predict
	predictions = model.predict(xMatrix)

	def f(q):
		# Construct feature matrix
		xMatrix = np.subtract.outer(q,x)
		xMatrix[xMatrix <= 0] = 0
		xMatrix = np.hstack((xMatrix,xMatrix**2))

		# Predict
		predictions = model.predict(xMatrix)

		return predictions

	return predictions, model, range(len(x)), f

import numpy as np
from sklearn.linear_model import Lasso as Lasso #LassoLarsIC as Lasso

def lassoFitLinear(x, y):
	# Sort
	ind = np.argsort(x)
	xx = x[ind]
	yy = y[ind]

	x = xx
	y = yy

	# Construct feature matrix
	xMatrix = np.subtract.outer(x,x)
	xMatrix[xMatrix <= 0] = 0

	# Fit
	model = LassoCV(max_iter=100000, verbose=True, n_jobs=-1, normalize=True, fit_intercept=True)
	model.fit(xMatrix,y)

	print(model.alpha_, np.sum(np.abs(model.coef_)>0),model.coef_)

	# Predict
	predictions = model.predict(xMatrix)

	def f(q):
		# Construct feature matrix
		xMatrix = np.subtract.outer(q,x)
		xMatrix[xMatrix <= 0] = 0

		# Predict
		predictions = model.predict(xMatrix)

		return predictions

	return predictions, model, range(len(x)), f