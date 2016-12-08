from __future__ import absolute_import

from keras import backend as K
from keras import initializations, regularizers, constraints
from keras.engine import Layer
from keras_dt import *

def indeces_trees(trees):
    return np.unique(trees,return_inverse=True)[1]  


class EmbeddingDT(Layer):
    
    input_ndim = 2
    
    def __init__(self,dt, trees, limit, input_dim, output_dim,
                  input_length=None
                 , **kwargs):
        self.input_dim = input_dim
        self.dt = dt
        self.output_dim = output_dim
        self.input_length = input_dim
        self.cache = []
        self.trees = trees
        self.limit = limit

        kwargs['input_shape'] = (self.input_length,)
        kwargs['trainable'] = False
        super(EmbeddingDT, self).__init__(**kwargs)

    def build(self, input_shape):
    	#ci serve la matrice identita'
        self.W = K.eye(self.output_dim)

    def get_output_shape_for(self, input_shape):
        if not self.input_length:
            input_length = input_shape[1]
        else:
            input_length = self.input_length
        #print (input_shape[0], input_length, self.output_dim)
        return (input_shape[0], input_length, self.output_dim)
    
    def call(self, x, mask=None):

        if type(x) != np.int64:
            return K.zeros((1,))
        if len(self.cache) < self.limit:

            self.cache.append(self.dt.dt(self.trees[x]))
            return self.cache[-1]
        return K.dot(W, self.dt.dt(self.trees[x]))


    
