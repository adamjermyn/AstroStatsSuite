import numpy as np
import sys
sys.path.append('../Fit Code/')

from funcs import *
from ZeBRA4 import fit as ourFit
from medians import medianFit
from mars import marsFit
from boosted import boostTreeFit
from lassoFit import lassoFit, lassoFitLinear
from kriging import krigingFit

def findBias(modelFunc, noiseFunc, fitFunc, xMin, xMax, num, args, giveErrors):
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
	bias = 0

	# Create data
	x = np.linspace(-3,3,num=num)
#			x = np.random.rand(num)*(xMax - xMin) + xMin

	# Compute underlying function plus noise
	y = modelFunc(x) + noiseFunc(x)
	errs = np.ones(len(x))
	# Fit model
	# predictions holds the fitted y
	# params holds the parameters of the fit model
	# indices specifies which values of x have predictions
	# (for models which don't return predictions everywhere)
	# giveErrors specifies whether or not the fit function accepts the keyword argument errors, giving the variance everywhere
	if len(args) > 0:
		if giveErrors:
			predictions, params, indices, model = fitFunc(x,y,errors=errs,*args)	
		else:
			predictions, params, indices, model = fitFunc(x,y,*args)	
	else:
		if giveErrors:
			predictions, params, indices, model = fitFunc(x,y,errors=errs)	
		else:
			predictions, params, indices, model = fitFunc(x,y)	


	bias = (np.sum(predictions[np.where((x>-0.9) & (x<0.9))])/len(x[np.where((x>-0.9) & (x<0.9))]) - 10)/10

	# Compute true bias
	#biasTrue = np.sum(predictions - modelFunc(x)[indices])/len(predictions)
	# Accumulate results

	return bias


fitNames = ["Medians","Ours with errors","Ours","Kriging","BoostTree","MARS"]
fitFuncs = [medianFit,ourFit,ourFit, krigingFit, boostTreeFit,marsFit]
fitArgs = [(5,),tuple(),tuple(),tuple(),tuple(),tuple()]
giveErrors = [False, True, False, False, False, False]

# Make the starting point depend on the fit method

numTrials = [400,400,400,400,400,400]
n = [200,200,200,200,200,200]
biases = []

for i in range(len(fitFuncs)):
	biases.append([])
	for k in range(20,n[i],4):
		print(i,k)
		for q in range(numTrials[i]):
			bias = findBias(lightCurve,gaussianNoise,fitFuncs[i],-2,2,k,fitArgs[i],giveErrors[i])
			biases[-1].append((k,bias))

print(biases)
fi = open('../Benchmark Output/findNtransit.out','wb')
import pickle
pickle.dump(biases,fi)
