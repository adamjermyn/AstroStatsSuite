import numpy as np

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