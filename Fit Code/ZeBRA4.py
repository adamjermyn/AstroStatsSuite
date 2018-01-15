import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from statsmodels.regression.linear_model import WLS
import statsmodels.api as sm

def wls(x):
	'''
	Performs weighted least squares on the numpy array x of shape
	(N,3), where the three components are indep variable, dep variable,
	dep variable errors. Returns log(likelihood).
	'''
	wls_model = WLS(x[:,1], sm.add_constant(x[:,0]), weights=1/x[:,2]**2)
	result = wls_model.fit()
	return -np.sum((x[:,1] - wls_model.predict(result.params))**2/x[:,2]**2), result.params[1], x[:,1] - wls_model.predict(result.params)

def errorEstimate(x, y):
	width = int(np.sqrt(len(x)))
	ms = np.zeros(len(y))
	counter = np.zeros(len(y))
	for i in range(width, len(ms) - width + 1):
		xx = x[i - width:i + width]
		yy = y[i - width:i + width]
		errs = np.ones(len(xx))
		resid = wls(np.transpose(np.vstack((xx,yy,errs))))[2]
		ms[i - width:i + width] += np.average(resid**2)
		counter[i - width:i + width] += 1
	ms /= counter
	ms = np.sqrt(ms)
	return ms

def pvalExp(n):
	k = n - 2
	return -1.0*k*(1-2/(9*k))**3

def split(knot, cutoff):
	'''
	Takes as input a knot and a cutoff and either returns
	the knot or splits it and calls itself again on the two halves.
	'''
	if len(knot) <= 8:
		return [knot]
	p, length, _ = wls(knot)
	pexp = pvalExp(len(knot))
	if p > pexp:
		return [knot]
	else:
		# Split
		best = [-1e50,0]
		for i in range(4,len(knot)-4):
			p1, l1, _ = wls(knot[:i])
			p2, l2, _ = wls(knot[i:])
			dx1 = np.std(knot[:i,0])
			dx2 = np.std(knot[i:,0])
			p = p1 + p2 - pvalExp(i) - pvalExp(len(knot)-i)
			if p > best[0]:
				best[0] = p
				best[1] = i
		return split(knot[:best[1]],cutoff) + split(knot[best[1]:],cutoff)

def rawFit(x, y, errors):
	# Initialize knots. 
	# We start with a single knot containing all points.
	# We then iteratively ask the question "Are the points in
	# this knot consistent with being a line?" If the answer is yes
	# then we do nothing to that knot. If the answer is no we identify
	# the break point inside the knot with the highest p-value and
	# recurse.
	knot = np.transpose(np.vstack((x, y, errors)))
	knots = split(knot, np.log(0.5))

	if len(knots) > 2:
		knotLocs = [knots[0][1][0]] + [k[0][0] for k in knots[1:-1]] + [knots[-1][-2][0]]
	elif len(knots) == 2:
		knotLocs = [knots[0][1][0],knots[0][-1][0],knots[1][-2][0]]
	else:
		knotLocs = [knots[0][1][0],knots[0][-2][0]]

	return knots, knotLocs

def fit(x, y, errors=None):
	# Sort
	ind = np.argsort(x)
	x = x[ind]
	y = y[ind]

	# Fill in errors
	if errors is None:
		errors = errorEstimate(x, y)

#	print(np.vstack((x,y,errors)).T)

	# We learn a global multiplicative factor on top of the errors
	# using cross-validation. This lets us set the cutoff to the most
	# computationally efficient value and removes a degree of freedom.
	cutoff = np.log(0.5)

	alphas = 10**np.linspace(-3.,3.,num=21,endpoint=True)

	best = [1e100,alphas[len(alphas)//2]]

	for a in alphas:
		# sub-sample for cross-validation
		inds = np.random.choice(len(x), size=len(x)//5,replace=False)
		xS = x[inds]
		yS = y[inds]
		errorsA = a*errors[inds]
		inds = np.argsort(xS)
		xS = xS[inds]
		yS = yS[inds]
		errorsA = errorsA[inds]

		knots, knotLocs = rawFit(xS, yS, errorsA)
		interp = LSQUnivariateSpline(xS, yS, knotLocs, k=1, w=1/errorsA**2)
		resid = np.sum((interp(x) - y)**2)
#		print(best,a,resid)
		if resid < best[0]:
			best[0] = resid
			best[1] = a

	knots, knotLocs = rawFit(x, y, errors*best[1])
	interp = LSQUnivariateSpline(x, y, knotLocs, k=1, w=1/errors**2)

	print('Final Answer:',len(knots))

	return interp(x), (x, y, knots, knotLocs, best), range(len(x)), interp


def fitWithErrors(x, y, errors):
	return fit(x, y, errors=errors)


#from scipy.signal import square
#x = np.linspace(0,5,num=300,endpoint=True)
#y = np.random.randn(len(x)) + 10*square(2*x)
#fit(x, y)

x = np.random.rand(100)
y = np.random.randn(len(x))*np.sqrt(x) + x
pred, params, indices, interp = fit(x, y)
print(params[-1])
pred, params, indices, interp = fit(x, y, errors=np.sqrt(x))
print(params[-1])


