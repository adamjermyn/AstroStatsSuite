import matplotlib
matplotlib.use('agg')
#matplotlib.use('svg')
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
import numpy as np
import math
import csv
import sys
sys.path.append('../Fit Code/')
from ourFit import ourFit
from mars import marsFit
from boosted import boostTreeFit

#plt.xkcd()

#matplotlib.rcParams['text.usetex']=True
#matplotlib.rcParams['text.latex.unicode']=True

csv.register_dialect('ssv', delimiter=' ', skipinitialspace=True)

plt.cla()
plt.clf()
plt.close()

#plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)

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
plt.scatter(m,l)

postfix = [' (x)',' (y)']
fitNames = ["Boosted Trees","MARS","ZeBRA"]
fitFuncs = [boostTreeFit, marsFit, ourFit]
fitArgs = [tuple(),tuple(),(0.05,)]
x,y = m,l
x = x[::100]
y = y[::100]
x = np.array(x)
y = np.array(y)

for i in range(len(fitFuncs)):
    for j in range(len(postfix)):
        if j==0:
            inds = np.argsort(x)        
            x = x[inds]
            y = y[inds]
            predictions, params, indices, model = fitFuncs[i](x, y, *fitArgs[i])
            plt.plot(x[indices], predictions, label=fitNames[i] + postfix[j], linewidth=4)
        else:
            inds = np.argsort(y)        
            x = x[inds]
            y = y[inds]
            predictions, params, indices, model = fitFuncs[i](y, x, *fitArgs[i])
            plt.plot(predictions, y[indices], label=fitNames[i] + postfix[j], linewidth=4)

meds = []
mx = []
my = []
with open('Quasar Data/mx.txt', 'rt') as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
        floats = [float(column) for column in row]
        meds.append(floats)
    for row in meds:
        mx.append(row[0])
        my.append(row[1])
l1 = plt.plot(mx,my, label='Medians (y)', linewidth=4)


meds = []
mx = []
my = []
with open('Quasar Data/my.txt', 'rt') as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
        floats = [float(column) for column in row]
        meds.append(floats)
    for row in meds:
        mx.append(row[0])
        my.append(row[1])
l2 = plt.plot(mx,my, label='Medians (x)', linewidth=4)


plt.legend(bbox_to_anchor=(0.4, 0.91),
           bbox_transform=plt.gcf().transFigure)
plt.savefig("../Plots/quasarAll.pdf", bbox_inches='tight')
plt.show()
