import numpy as np
from scipy.signal import square as sq

def constFunc(x):
	return np.ones(len(x))

def logFunc(x):
	return np.log(1+x)

def expFunc(x):
	return np.exp(x/3)

def schechterFunc(x):
	return (x+1)**(-0.46)*np.exp(-(x+1))

def piecewiseFunc1(x):
	return 2*x + 4*(x-5.0)*(x>5.0)

def curvedFunc1(x):
	return x**2

def curvedFunc2(x):
	return x**3

def uniformNoise(x):
	return np.random.rand(len(x)) - 0.5

def gaussianNoise(x):
	return np.random.randn(len(x))

def largeGaussianNoise(x):
	return 10*np.random.randn(len(x))

def relativeGaussianNoise(x):
	return 0.33*np.random.randn(len(x))

def logNormalNoise(x):
	return np.random.lognormal(size=len(x))

def lightCurve(x):
	y = np.zeros(len(x))
	y[x<-1] = 0
	y[x>1] = 0
	y[(x>=-1) & (x <= -0.9)] = 10*(x[(x>=-1) & (x <= -0.9)]+1)
	y[(x>=-0.9) & (x <= 0.9)] = 1
	y[(x>=0.9) & (x <= 1)] = 10*(1-x[(x>=0.9) & (x <= 1)])
	return 10*y - 3

def curved(x):
	k = 2*np.pi
	return 4 + np.sin(k*x) + np.sin(2*k*(x+0.1)) + np.cos(3*k*x) + np.cos(7*k*x)

def linear(x):
	return x

def square(x):
	return 4 + sq(2*np.pi*5*x, duty=0.5)
