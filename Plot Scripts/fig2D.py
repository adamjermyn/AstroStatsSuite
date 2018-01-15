import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import sys
sys.path.append('../Fit Code/')
import deconFit

csv.register_dialect('ssv', delimiter=' ', skipinitialspace=True)

plt.cla()
plt.clf()
plt.close()

#plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

plt.xlim(7.5,10.5)
plt.ylim(45,48)
plt.xlabel(r'Log Black Hole Mass ($M_\odot$)')
plt.ylabel(r'Log Luminosity ($L_\odot$)')

m = []
l = []

with open('Quasar Data/pointsm.txt', 'rt') as f:
	reader = csv.reader(f)
	for row in reader:
		floats = [float(column) for column in row]    
		m.append(floats[0])

with open('Quasar Data/pointsl.txt', 'rt') as f:
	reader = csv.reader(f)
	for row in reader:
		floats = [float(column) for column in row]    
		l.append(floats[0])       

m = np.array(m)
l = np.array(l)

plt.scatter(m,l)

# Note that doing this means that we end up lumping all of the scatter in
# the observation and not in the underlying distribution. Unless we have a
# prior on what these ought to be this is the best we can do...

inds = np.argsort(m)
m = m[inds]
l = l[inds]

# There are 16 bad datapoints at the beginning after sorting.
m = m[16:]
l = l[16:]

m = m[::20]
l = l[::20]

dx, dy = deconFit.bandwidthEstimate(m, l)
print(dx, dy)

n, clf, grid, minX, maxX, minY, maxY = deconFit.findNumComponents(m, l, dx, dy)

print(n)

assert n == 1

vals, vects = np.linalg.eig(clf.V[0])
print(vals)
print(vects)
if abs(vals[0]) > abs(vals[1]):
	vect = vects[:,0]
else:
	vect = vects[:,1]

print(vect[1]/vect[0])

mRan = np.linspace(min(m),max(m),num=1000)
plt.plot(mRan, (mRan-clf.mu[0][0])*vect[1]/vect[0] + clf.mu[0][1], linewidth=2,c='r',label='Deconvolution Fit')

plt.legend(bbox_to_anchor=(0.55, 0.89),
           bbox_transform=plt.gcf().transFigure)

plt.savefig('../Plots/gaussianFit.pdf')

