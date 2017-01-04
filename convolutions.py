from keras import backend as K
import numpy as np
from vectors import *
from scipy.linalg import circulant
# according this review http://scottsievert.com/blog/2016/07/01/numpy-gpu/
# numpy fast computes FFT without acceleration (theano use numpy for fft)

#circular convolution, x,y arrays, permutations is not used
def circular_convolution(x,y,permutations=None):
	if type(x) != np.ndarray:
		x=x.eval()
	if type(y) != np.ndarray:
		y=y.eval()
	return np.real(np.fft.ifft(np.fft.fft(x)*np.fft.fft(y)))
#circular convolution, x,y arrays, permutations is used
def shuffled_circular_convolution(x,y,permutations=None):
	if type(x) != np.ndarray:
		x=x.eval()
	if type(y) != np.ndarray:
		y=y.eval()
	
	return circular_convolution(x[permutations[0]],y[permutations[1]],permutations)

def permutation_matrices(N):
	I = np.eye(N)
	p1 = circulant(I[:,0])

	return p1, p1.T

def cc_circulant(x,y):
	A = circulant(x) #circulant matrix
	#permuation matrices
	Phi1,Phi2 = permutation_matrices(x.shape[0])
	#circular convolution
	#print Phi1,Phi2
	return Phi1.dot(A).dot(Phi2).dot(y)
if __name__ == '__main__':
	x=np.array([1,2,3])
	y=np.array([2,3,4])
	print cc_circulant(x,y), circular_convolution(x,y)
