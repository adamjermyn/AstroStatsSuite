import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append('../Fit Code/')
#from boosted import boostTreeFit as fit
#from mars import marsFit as fit
#from medians import medianFit as fit
from ZeBRA4 import fit

plt.rc('font', family='serif', size=15)

fitNames = ["Medians","ZeBRA","Kriging","BoostTree","MARS"]

data = pickle.load(open('../Benchmark Output/findNtransit.out','rb'))

for i in range(len(data)):
	data[i] = np.array(data[i])

data2 = []
for i in range(len(data)):
	if i != 1:
		data2.append([])
		for k in range(20,200,4):
			pts = data[i][data[i][:,0]==k,1]
			pts *= 10 # Because sigma=0.1
			pts += 3 # To handle the offset in funcs
			data2[-1].append([k,(pts**2).mean()**0.5,np.std(pts)/np.sqrt(len(pts))])
		data2[-1] = np.array(data2[-1])

data = data2

xRan = np.linspace(20,200,num=200,endpoint=True)

for i in range(len(data)):
	_, _, _, model = fit(data[i][:,0],data[i][:,1],errors=data[i][:,2])
	plt.semilogy(xRan, model(xRan),label=fitNames[i],linewidth=2)

plt.xlim([0,200])
plt.xlabel('N')
plt.ylabel('Log Bias ($\sigma$)')

plt.legend()
plt.legend(bbox_to_anchor=(0.9, 0.9),
           bbox_transform=plt.gcf().transFigure)
# Need to move the legend over

plt.savefig('../Plots/findNtransit.pdf')
