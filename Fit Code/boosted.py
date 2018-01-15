import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

def boostTreeFit(x,y):
	est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls').fit(x[:,np.newaxis],y)

	def f(x):
		return est.predict(x[:,np.newaxis])

	return est.predict(x[:,np.newaxis]),est,range(len(x)),f
