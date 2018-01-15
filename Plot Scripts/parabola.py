from matplotlib.patches import Ellipse
from astroML.plotting.tools import draw_ellipse
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../Fit Code/')
import deconFit
from hermiteSpline import interpolator
from widestPath import path

def parabolaTest(n):
	x = 4*np.random.randn(n)
	y = x**2 + 0.2*np.random.randn(n)
	x += 0.2*np.random.randn(n)

	dx = 0.2*np.ones(len(x))
	dy = 0.2*np.ones(len(y))

	n, clf, grid, minX, maxX, minY, maxY = deconFit.findNumComponents(x, y, dx, dy)

	fig = plt.figure(figsize=(4,7))
	ax = fig.add_subplot(211)

	for i in range(clf.n_components):
		draw_ellipse(clf.mu[i], clf.V[i], scales=[2], ax=ax,
				 ec='k', fc='gray', alpha=0.5)

	pts = clf.mu
	covars = clf.V


	pts = np.transpose(pts)

	vals, vect = np.linalg.eig(covars)
	inds = np.argmax(np.abs(vals), axis=1)
	vects = []
	for i in range(len(vect)):
		vects.append(vect[i,:,inds[i]])
	vect = np.array(vects)

	pth = path(pts, covars, clf.alpha, vertical=False)

	t = np.array(list(range(len(pth))))
	tRan = np.linspace(min(t),max(t),num=200)
	xRan = interpolator(t, pts[0,pth], vect[:,0], tRan)
	yRan = interpolator(t, pts[1,pth], vect[:,1], tRan)

	print(vect)

	plt.rc('font', family='serif', size=15)
	plt.scatter(x,y,c='k')
	plt.plot(xRan, xRan**2, label='$Y=X^2$',c='r')
	plt.plot(xRan, yRan, label='Fit')
	plt.xlabel('X')
	plt.ylabel('Y')

	plt.legend()

	ax = fig.add_subplot(212)

	for i in range(clf.n_components):
		draw_ellipse(clf.mu[i], clf.V[i], scales=[2], ax=ax,
				 ec='k', fc='gray', alpha=0.2)

	pth = np.array(pth)

	pth = pth[np.argsort(clf.mu[pth,1])]

	t = np.array(list(range(len(pth))))
	tRan = np.linspace(min(t),max(t),num=200)
	xRan = interpolator(t, pts[0,pth], vect[:,0], tRan)
	yRan = interpolator(t, pts[1,pth], vect[:,1], tRan)

	print(vect)

	plt.scatter(x,y,c='k')
	plt.plot(xRan, xRan**2, label='$Y=X^2$',c='r')
	plt.plot(xRan, yRan, label='Fit')
	plt.xlabel('X')
	plt.ylabel('Y')

	plt.savefig('../Plots/parabola.pdf')

parabolaTest(100)
