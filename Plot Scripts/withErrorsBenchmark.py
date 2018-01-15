import time
import numpy as np
import sys
sys.path.append('../Fit Code/')

from funcs import *
from ourFit import ourFit
from ZeBRA4 import fitWithErrors as fit
from medians import medianFit, binnedMedianFit
from mars import marsFit
from boosted import boostTreeFit
from lassoFit import lassoFit, lassoFitLinear
from kriging import krigingFit
from means import meanFit
from connectTheDots import ctdFit

def trial(modelFunc, noiseFunc, fitFunc, xMin, xMax, num, numTrials, args):
	'''
	modelFunc - Underlying function generating the data
	noiseFunc - Noise that we add on top of the underlying function
	fitFunc   - Function we use to fit the data
	xMin	  - Left end of the domain
	xMax	  - Right end of the domain
	num	  - Number of points to use per trial
	numTrials - Number of trials
	args	  - Tuple of arguments to pass to the fitting function after x and y
	'''

	netBiasTrue = 0.
	netVarianceTrue = 0.
	netTime = 0.

	for i in range(numTrials):
		# Create data
		x = np.random.rand(num)*(xMax - xMin) + xMin
		inds = np.argsort(x)
		x = x[inds]

		# Compute underlying function plus noise
		y = modelFunc(x) + np.abs(modelFunc(x))**0.5*noiseFunc(x)
		errs = np.abs(modelFunc(x))**0.5
		#noiseFunc(x)*modelFunc(x)

		# Fit model
		start_time = time.clock()
		# predictions holds the fitted y
		# params holds the parameters of the fit model
		# indices specifies which values of x have predictions
		# (for models which don't return predictions everywhere)

		predictions, params, indices, model = fitFunc(x,y, errs)	
					
		elapsed_time = time.clock()-start_time

		# Compute true bias
		ran = xMax - xMin
		xx = np.linspace(xMin + 0.25*ran,xMax - 0.25*ran,num=400,endpoint=True)
		predictions = model(xx)
		yy = modelFunc(xx)
		biasTrue = np.sum((predictions - yy)/yy**0.5)/len(predictions)
		varianceTrue = np.sum(((predictions - yy)**2)/yy)/len(predictions)

		# Accumulate results
		netBiasTrue += biasTrue
		netVarianceTrue += varianceTrue
		netTime += elapsed_time

	print('Range:',xMin, xMax)
	print('Average true bias: ', netBiasTrue / numTrials)
	print('Average true variance: ', netVarianceTrue / numTrials)
	print('Average time: ', netTime / numTrials)

	return np.array([netBiasTrue, netVarianceTrue, netTime])/numTrials

modelNames = ["Sines", "Linear", "Square"]
modelFuncs = [curved, linear, square]
noiseNames = ["Gaussian"]
noiseFuncs = [gaussianNoise]
fitNames = ['ZeBRA']
fitFuncs = [fit]
fitArgs = [tuple()]

nPts = 100
nTrials = 100
xMin = 0
xMax = 1.0

print('Beginning Benchmarks with')
print('Number of Observations: ',nPts)
print('Number of Trials: ',nTrials)
print('Range: [',xMin,',',xMax,']')
print('\n')

data = []

for i in range(len(modelFuncs)):
	for j in range(len(noiseFuncs)):
		for k in range(len(fitFuncs)):
			print('-------------')
			print("Model: ",modelNames[i])
			print("Noise: ",noiseNames[j])
			print("Fit: ",fitNames[k])
			data.append(trial(modelFuncs[i],noiseFuncs[j],fitFuncs[k],xMin,xMax,nPts,nTrials,fitArgs[k]))
			print('\n')

np.savetxt('../Benchmark Output/benchmarksSE.out',np.array(data))

fitNames = ["ZeBRA"]
fitFuncs = [fit]
fitArgs = [tuple()]

nPts = 10000
nTrials = 100
xMin = 0
xMax = 1.0

print('Beginning Benchmarks with')
print('Number of Observations: ',nPts)
print('Number of Trials: ',nTrials)
print('Range: [',xMin,',',xMax,']')
print('\n')

data = []

for i in range(len(modelFuncs)):
	for j in range(len(noiseFuncs)):
		for k in range(len(fitFuncs)):
			print('-------------')
			print("Model: ",modelNames[i])
			print("Noise: ",noiseNames[j])
			print("Fit: ",fitNames[k])
			data.append(trial(modelFuncs[i],noiseFuncs[j],fitFuncs[k],xMin,xMax,nPts,nTrials,fitArgs[k]))
			print('\n')

np.savetxt('../Benchmark Output/benchmarksME.out',np.array(data))
