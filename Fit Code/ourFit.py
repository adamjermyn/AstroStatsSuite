import numpy as np
from scipy.interpolate import LSQUnivariateSpline

class fitfunc:
	def __init__(self, x, y, errors=None):
		if errors is not None:
			self.errors = errors
		else:
			grad = np.gradient(y)
			ms = grad**2
			ms2 = np.zeros(len(ms)+4)
			ms2[0] = ms[0]
			ms2[1] = ms[0]
			ms2[2:-2] = ms
			ms2[-2] = ms[-1]
			ms2[-1] = ms[-1]
			ms = ms2
			ms = np.convolve(ms, np.ones((5,))/5, mode='valid')
			ms /= 2 # Correction for difference of uncorrelated errors
			self.errors = np.sqrt(ms)
			self.errors = np.copy(np.sqrt(np.abs(y)))

		self.weights = self.errors**2

		# Save inputs and enforce sorting along x
		inds = np.argsort(x)
		self.x = np.copy(x)[inds] + 1e-10*np.array(range(len(x))) # This prevents duplicates in x
		self.y = np.copy(y)[inds]

		# Setup knots
		self.knots = np.copy(self.x[1:-1])
		self.knotInd = np.array(range(len(self.knots)))
		self.knotPvals = None

		# Iteratively fit and remove knots
		self.interp = None
		self.fit()
		self.pvals()
		done = False
		while not done and len(self.knots) > 2:
			choice = np.random.choice(range(len(self.knots)),len(self.knots),replace=False)
			for i in choice:
				if self.knotPvals[i] < 0:
					self.removeKnot(i)
					self.fit()
					self.pvals()
					break
			else:
				done = True

		self.interpDeriv = self.interp.derivative(1)
		print('Converged with ',len(self.knots),'knots.')
		print(self.knotPvals,self.interp.get_residual())

	def fit(self):
		self.knotPvals = np.zeros(len(self.knots))
		self.interp = LSQUnivariateSpline(self.x,self.y,self.knots,k=1, w=self.weights)
		
	def pvals(self):
		for i in range(len(self.knots)):
			self.pval(i)

	def pval(self, index):
		### In practice we'd implement some actual error scheme for this
		knotsDel = np.delete(self.knots,index)
		interpDel = LSQUnivariateSpline(self.x,self.y,knotsDel,k=1,w=self.weights)

		resid = self.interp.get_residual()
		residDel = interpDel.get_residual()

		dof = len(self.x) - len(self.knots) - 2 # -2 because there are 2 parameters in a 1-knot k=1 fit.
		dofdel = dof + 1

		if dof == 0:
			self.knotPvals[index] = -np.inf
		else:
			self.knotPvals[index] = abs(residDel/dofdel - 1) - abs(resid/dof - 1)
			print(self.knotPvals[index], resid/dof, residDel/dofdel, resid, dof)

	def removeKnot(self, index):
		# Note that index can be a list of indices, or even
		# a smart-indexing tool.
		self.knots = np.delete(self.knots,index)
		self.knotInd = np.delete(self.knotInd,index)
		self.slope = None
		self.intercept = None
		self.knotPvals = None

	def evalY(self, x):
		return self.interp(x)

	def evalDeriv(self, x):
		return self.interpDeriv(x)

def ourFit(x, y, errors):
	f = fitfunc(x, y, errors=errors)
	return f.evalY(x), (f.interp, f.interpDeriv), range(len(x)), f.interp
