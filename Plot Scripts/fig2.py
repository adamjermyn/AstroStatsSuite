import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

import sys
sys.path.append('../Fit Code/')

from medians import medianFit
from means import meanFit
from funcs import gaussianNoise

n = 100

#def lightFunc(x):
#	y  = x/(10**10.93)
#	return 1.11*(y**(-0.46))*np.exp(-y)
from funcs import lightCurve
def lightFunc(x):
	return 20 - lightCurve(x)

x = np.random.randn(n)
#x = 10**(np.random.randn(n)*1 + 10.5)

#x = x[np.where((x>10**8) & (x<10**12))]

y = lightFunc(x) + gaussianNoise(x)

#xRan = 10**np.linspace(min(np.log10(x)),max(np.log10(x)),num=1000)
xRan = np.linspace(min(x),max(x),num=1000)
plt.rc('font', family='serif', size=15)

fig1 = plt.figure(1)

frame1 = fig1.add_axes((0.1,0.3,0.8,0.6))

#plt.xlabel('log(M)[$M_\odot$]')
#plt.ylabel('log($\phi$)($Mpc^{-3}dex^{-1}$)')

plt.xlabel('Time')
plt.ylabel('Flux')

#plt.plot(np.log10(xRan), np.log10(medianModel(xRan)),c='orange',label='Running Medians', linewidth=2)
#plt.plot(np.log10(xRan), np.log10(meanModel(xRan)),c='r',label='Running Means', linewidth=2)
#plt.plot(np.log10(xRan), np.log10(lightFunc(xRan)),c='b',label='Schechter Function',linewidth=2)
#plt.scatter(np.log10(x),np.log10(y),c='k')

yM1, _, _, medianModel = medianFit(x, y, 5)
yM2, _, _, meanModel = meanFit(x, y, 5)

plt.scatter((x),(y),c='k')

plt.plot((xRan), (medianModel(xRan)),c='orange',label='Running Medians', linewidth=2)
plt.plot((xRan), (meanModel(xRan)),c='r',label='Running Averages', linewidth=2)
plt.plot((xRan), (lightFunc(xRan)),c='b',label='Flux',linewidth=2)

plt.xlim([-0.5+min(x),0.5+max(x)])

plt.ylim([-10+min(y),2+max(y)])

frame1.set_xticklabels([]) #Remove x-tic labels for the first frame

plt.legend(bbox_to_anchor=(0.54, 0.55),
           bbox_transform=plt.gcf().transFigure)

frame2=fig1.add_axes((.1,.1,.8,.2))        

plt.xlabel('Time')
plt.ylabel('log Relative Error')

plt.locator_params(axis='y',nbins=5)

plt.plot(xRan, np.log10(medianModel(xRan)/lightFunc(xRan)),c='orange',label='Running Medians', linewidth=2)
plt.plot(xRan, np.log10(meanModel(xRan)/lightFunc(xRan)),c='r',label='Running Averages', linewidth=2)

plt.xlim([-0.5+min(x),0.5+max(x)])

plt.savefig('../Plots/fig2.pdf',bbox_inches='tight')