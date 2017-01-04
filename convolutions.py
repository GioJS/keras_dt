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

def cc_circulant(x,y):
	A = circulant(x) #circulant matrix
	#permuation matrices
	Phi_1 = []
	Phi_2 = []
	#circular convolution
	return Phi_1 * A * Phi_2 * y