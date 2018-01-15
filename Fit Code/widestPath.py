import numpy as np
import networkx as nx

def bhattacharyyaDistance(mu1, mu2, covar1, covar2):
	covar = (covar1 + covar2)/2

	covarInv = np.linalg.inv(covar)

	d = np.einsum('i,ij,j',mu1-mu2,covarInv,mu1-mu2)/8

	d += 0.5*np.log(np.linalg.det(covar)/np.sqrt(np.linalg.det(covar1)*np.linalg.det(covar2)))

	return d

def hellingerDistance(mu1, mu2, covar1, covar2):
	return np.sqrt(1 - bhattacharyyaDistance(mu1, mu2, covar1, covar2))

def path(pts, covars, weights, vertical=False):
	'''
	This method returns the path (as a list of indices) through
	pts which maximizes the minimum probability over the links
	of the paths.

	This function takes three arguments as inputs:
		pts 		-	The points of interest as a (2,N) array.
		covars		-	The covariance matrix for each point as a (2,2,N) array.
		weights		-	The weights to assign to each point. Currently not used.
		vertical 	-	Default is False. If False uses the min and max points in
						x (the first dimension) as starting and ending points.
						If True uses the min and max in y (the second dimension).
	'''

	x, y = pts

	dists = np.zeros((len(x),len(x)))

	for i in range(len(x)):
		for j in range(len(y)):
			if i != j:
				dists[i,j] = bhattacharyyaDistance(pts[:,i], pts[:,j], covars[i], covars[j])
#				dists[i,j] += 0.5*np.log(weights[i])
#				dists[i,j] += 0.5*np.log(weights[j])


	keepIndices = []

	for i in range(len(x)):
		keep = False
		for j in range(len(y)):
			if dists[i,j] < 5:
				keep = True
		if keep:
			keepIndices.append(i)

	x = x[keepIndices]
	y = y[keepIndices]

	if vertical:
		start = np.argmin(y)
		stop = np.argmax(y)
	else:
		start = np.argmin(x)
		stop = np.argmax(x)

	G = nx.Graph()

	for i in range(len(x)):
		G.add_node(i)


	for i in range(len(x)):
		for j in range(len(y)):
			if i!=j:
				G.add_edge(i,j,weight=np.exp(dists[i][j]))

	t = nx.minimum_spanning_tree(G,weight='weight')
	pth = nx.shortest_path(t, start, stop)

	return pth