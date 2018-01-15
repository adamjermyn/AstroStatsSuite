import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

import sys
sys.path.append('../Fit Code/')

from ZeBRA4 import fit as ourFit
from medians import medianFit
from mars import marsFit
from boosted import boostTreeFit
from lassoFit import lassoFit
from kriging import krigingFit
from means import meanFit
from funcs import gaussianNoise

n = 200

from funcs import lightCurve
def lightFunc(x):
	return 13 - lightCurve(x)

x = np.random.randn(n)

y = lightFunc(x) + gaussianNoise(x)

yM1, _, _, medianModel = medianFit(x, y, 5)
yM2, _, _, meanModel = meanFit(x, y, 5)

xRan = np.linspace(min(x),max(x),num=1000)
plt.rc('font', family='serif', size=15)

fig1 = plt.figure(1,figsize=[5,7])

frame1 = fig1.add_axes((0.1,0.7,0.8,0.6))

plt.xlabel('Time')
plt.ylabel('Flux')

fitNames = ["BoostTree","MARS","Averages","Medians"]
fitFuncs = [boostTreeFit, marsFit, meanFit, medianFit]
fitArgs = [tuple(),tuple(),(5,),(5,)]
giveErrors = [False,False,False,False]
models = []
cols = ['r', 'g', 'b', 'y']

for i in range(len(fitNames)):
	if not giveErrors[i]:
		yM, _, _, model = fitFuncs[i](x, y, *(fitArgs[i]))
	else:
		yM, _, _, model = fitFuncs[i](x, y, errors=np.ones(len(x)),*(fitArgs[i]))

	plt.plot(xRan, model(xRan), label=fitNames[i],linewidth=2,c=cols[i])
	models.append(model)
plt.xlim([-1.3+min(x),0.5+max(x)])

plt.scatter((x),(y),c='k')

plt.ylim([-3,2+max(y)])
frame1.set_yticks([0,5,10,15,20])

frame1.set_xticklabels([]) #Remove x-tic labels for the first frame

plt.legend(bbox_to_anchor=(0.9, 0.85),
           bbox_transform=plt.gcf().transFigure,ncol=2)

frame2=fig1.add_axes((.1,.1,.8,.15))        


plt.locator_params(axis='y',nbins=5)

i = 0
plt.plot(xRan, 1 - models[i](xRan)/lightFunc(xRan),label=fitNames[i],linewidth=2,c=cols[i])

plt.xlim([-1.3+min(x),0.5+max(x)])
plt.ylim([-0.3,0.3])
frame2.set_yticks([-0.2,0,0.2])
plt.xlabel('Time')

frame3=fig1.add_axes((.1,.25,.8,.15))        


plt.locator_params(axis='y',nbins=5)

i = 1
plt.plot(xRan, 1 - models[i](xRan)/lightFunc(xRan),label=fitNames[i],linewidth=2,c=cols[i])

plt.xlim([-1.3+min(x),0.5+max(x)])
plt.ylim([-0.3,0.3])

frame3.set_xticks([])
frame3.set_yticks([-0.2,0,0.2])


frame4=fig1.add_axes((.1,.4,.8,.15))        

plt.ylabel('Relative Error')

plt.locator_params(axis='y',nbins=5)

i = 2
plt.plot(xRan, 1 - models[i](xRan)/lightFunc(xRan),label=fitNames[i],linewidth=2,c=cols[i])

plt.xlim([-1.3+min(x),0.5+max(x)])
plt.ylim([-0.3,0.3])

frame4.set_xticks([])
frame4.set_yticks([-0.2,0,0.2])

frame5=fig1.add_axes((.1,.55,.8,.15))        


plt.locator_params(axis='y',nbins=5)

i = 3
plt.plot(xRan, 1 - models[i](xRan)/lightFunc(xRan),label=fitNames[i],linewidth=2,c=cols[i])

plt.xlim([-1.3+min(x),0.5+max(x)])
plt.ylim([-0.3,0.3])

frame5.set_xticks([])
frame5.set_yticks([-0.2,0,0.2])



plt.savefig('../Plots/fig4.pdf',bbox_inches='tight')
