import matplotlib
matplotlib.use('agg')
#matplotlib.use('svg')
import matplotlib.pyplot as plt
import numpy as np
import math
import csv

#plt.xkcd()

#matplotlib.rcParams['text.usetex']=True
#matplotlib.rcParams['text.latex.unicode']=True

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
plt.scatter(m,l,c='k')

plt.savefig("../Plots/fig1_talk_noMedians.pdf", bbox_inches='tight')
plt.savefig("../Plots/fig1_talk_noMedians.jpg", bbox_inches='tight')

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
l2 = plt.plot(mx,my, label='Medians (y)', color='red', linewidth=4)

plt.savefig("../Plots/fig1_talk_Medians.pdf", bbox_inches='tight')
