import tabulate
import numpy as np

dataS = np.loadtxt('../Benchmark Output/benchmarksS.out')
dataM = np.loadtxt('../Benchmark Output/benchmarksM.out')

dataB = np.loadtxt('../Benchmark Output/benchmarksBin.out')

dataS = np.abs(dataS)
dataM = np.abs(dataM)
dataB = np.abs(dataB)

dataS[:,2] = np.sqrt(dataS[:,2])
dataM[:,2] = np.sqrt(dataM[:,2])
dataB[:,2] = np.sqrt(dataB[:,2])

# Variance calculation was wrong. Should be mse - bias**2. To fix this:
# Variance = MSE - Bias**2

dataS[:,1] = dataS[:,2]**2 - dataS[:,0]**2
dataM[:,1] = dataM[:,2]**2 - dataM[:,0]**2
dataB[:,1] = dataB[:,2]**2 - dataB[:,0]**2

modelNames = ["Curved","Linear","Square"]
modelSymbs = [	'$y=4 + \sin(2\pi x) + \sin(4 \pi (x+0.1)) + \cos(6\pi x)+ \cos(14\pi x)$',\
				'$y=x$',"$y=\mathrm{sgn}(\sin(10\pi x))$"]

### Print Smooth Data

print('----------------Smooth Data------------------')

fitNames = ["Random Forest", "Kriging", "Cubic Spline", "Local Linear","Local Quadratic","Connect the Dots","Means","Medians","Binned Medians (10)","MARS","Boosted Trees",'ZeBRA']

data3 = []
data4 = []

for i in range(len(dataS)//len(fitNames)):
	data3.append(['',12345,12345,12345,12345])
	data2 = [list(d) for d in dataS[len(fitNames)*i:len(fitNames)+len(fitNames)*i]]
	data2 = [[fitNames[j]] + data2[j] for j in range(len(fitNames))]
	data4 = []
	for j in data2:
		data3.append(j)

s = tabulate.tabulate(data3,headers=['$B$','$\sigma^2$','RMSE','Time (s)'],tablefmt='latex',floatfmt="2.4f")

s = s.replace('12345.0000','')
for i in range(len(modelNames)):
	s = s.replace(r'                     &     &   &  \\',r'\hline\\\multicolumn{4}{c}{'+modelNames[i] + '\ '+ modelSymbs[i]+r'}\\\hline\\',1)

print(s)

### Print Model Data

print('----------------Model Data------------------')

fitNames = ["Random Forest", "Cubic Spline", "Local Linear","Local Quadratic","Connect the Dots","Means","Medians","Binned Medians (10)","MARS","Boosted Trees",'ZeBRA']
data3 = []
data4 = []

for i in range(len(dataM)//len(fitNames)):
	data3.append(['',12345,12345,12345,12345])
	data2 = [list(d) for d in dataM[len(fitNames)*i:len(fitNames)+len(fitNames)*i]]
	data2 = [[fitNames[j]] + data2[j] for j in range(len(fitNames))]
	data4 = []
	for j in data2:
		data3.append(j)

s = tabulate.tabulate(data3,headers=['$B$','$\sigma^2$','RMSE','Time (s)'],tablefmt='latex',floatfmt="2.4f")

s = s.replace('12345.0000','')
for i in range(len(modelNames)):
	s = s.replace(r'                     &     &   &  \\',r'\hline\\\multicolumn{4}{c}{'+modelNames[i] + '\ '+ modelSymbs[i]+r'}\\\hline\\',1)

print(s)

### Print Bin Data

print('----------------Bin Data------------------')

fitNames = ["3","4","5","6","10","25","50","100"]

data3 = []
data4 = []

for i in range(len(dataB)//len(fitNames)):
	data3.append(['',12345,12345,12345,12345])
	data2 = [list(d) for d in dataB[len(fitNames)*i:len(fitNames)+len(fitNames)*i]]
	data2 = [[fitNames[j]] + data2[j] for j in range(len(fitNames))]
	data4 = []
	for j in data2:
		data3.append(j)

s = tabulate.tabulate(data3,headers=['$B$','$\sigma^2$','RMSE','Time (s)'],tablefmt='latex',floatfmt="2.4f")

s = s.replace('12345.0000','')
for i in range(len(modelNames)):
	s = s.replace(r'     &       &            &  \\',r'\hline\\\multicolumn{4}{c}{'+modelNames[i] + '\ '+ modelSymbs[i]+r'}\\\hline\\',1)

print(s)
