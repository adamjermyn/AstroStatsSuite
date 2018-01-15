import numpy as np

def h00(t):
	return 2*t**3 - 3*t**2 + 1

def h10(t):
	return t**3 - 2*t**2 + t

def h01(t):
	return -2*t**3 + 3*t**2

def h11(t):
	return t**3 - t**2

def interpolator(x, y, yp, x0):
	inds = np.argsort(x)
	x = x[inds]
	y = y[inds]
	yp = yp[inds]
	bins = np.copy(x)
	bins[-1] += 1e-10

	inds = np.digitize(x0,bins=bins)
	inds -= 1

	inds[inds == len(x)-1] = len(x)-2

	t = (x0 - x[inds])/(x[inds+1] - x[inds])

	return h00(t)*y[inds] + h01(t)*y[inds+1] +\
			 (x[inds+1]-x[inds])*yp[inds]*h10(t)+\
			 (x[inds+1]-x[inds])*yp[inds+1]*h11(t)