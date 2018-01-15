from pyearth import Earth

def marsFit(x,y):
	model = Earth(max_degree=1)
	model.fit(x,y)

	def f(x):
		return model.predict(x)

	return model.predict(x), model, range(len(x)), f