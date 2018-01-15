import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append('../Fit Code/')
from boosted import boostTreeFit

plt.rc('font', family='serif', size=15)

fitNames = ["Medians","ZeBRA","Kriging","BoostTree","MARS"]

data = pickle.load(open('../Benchmark Output/findNcosmo.out','rb'))

for i in range(len(data)):
	data[i] = np.array(data[i])
	data[i][np.isnan(data[i])] = 0
data[2] *= 0
print(data)

xRan = np.linspace(10,50,num=200,endpoint=True)

for i in range(len(data)):
	print(data[i])
	_, _, _, model = boostTreeFit(data[i][:,0],data[i][:,1])
	plt.plot(xRan, model(xRan),label=fitNames[i],linewidth=2)

plt.xlabel('z')
plt.ylabel('$\log d (pc)$')

plt.legend(bbox_to_anchor=(0.85, 0.51),
           bbox_transform=plt.gcf().transFigure)

plt.savefig('../Plots/findNcosmo.pdf')
