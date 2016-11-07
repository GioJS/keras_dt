from keras import backend as K
import numpy as np
import sys
'''
random vector generator(versors)
'''
class Vector_generator:
	def __init__(self,seed=0,dim=1024,mu=0,va=1):
		self.seed=seed
		self.dim=dim
		self.mu=mu
		self.va=va
		self.cache={}
	
	def get_random_vector(self,label):
		if label in self.cache:
			return self.cache[label]
		#if seed is hash adjust it (can be negative)
		seed = hash(label)
		if seed < 0:
			seed += sys.maxsize
	  	#self.seed=0
		np.random.seed(self.seed)
		vect = np.random.normal(self.mu,self.va,self.dim)
		vect /= np.linalg.norm(vect,2)
		self.cache[label]=vect
		return K.variable(vect)
	@staticmethod
	def permutation(dim=1024,seed=0,permutation=None):
		
		np.random.seed(seed)
		perm=np.random.permutation(dim)
		while(not Vector_generator.test_permutation(perm,permutation)):
			perm=np.random.permutation(dim)
		return perm
	@staticmethod
	def test_permutation(perm1,perm2=None):
		basic=np.arange(perm1.shape[0])
		shuffled=basic.copy()

		for i in range(basic.shape[0]):
			#print i
			if (shuffled[perm1] == basic).all():
				#print 'f'
				return False
			if not perm2 is None:
				if (shuffled == basic[perm2]).all():
					return False
				if (basic == shuffled[perm2]).all():
					return False
		return True
	@staticmethod
	def permutations(dim=1024,seed=0):
		perm1 = Vector_generator.permutation(dim=dim,seed=seed)
		#print 'perm1'
		perm2 = Vector_generator.permutation(dim=dim,seed=seed,permutation=perm1)
		return (perm1,perm2)