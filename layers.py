

from keras import backend as K
from keras import initializations, regularizers, constraints
from keras.engine import Layer
from keras_dt import *

def indices_trees(trees):
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

        super(EmbeddingDT, self).build(input_shape)

    def get_output_shape_for(self, input_shape):
       return (1,self.output_dim)
    
    def call(self, x, mask=None):

        if K.is_keras_tensor(x):
            return K.zeros((self.output_dim,))

        if x-1 < len(self.cache):
            return self.cache[x-1]

        if len(self.cache) < self.limit:
            self.cache.append(self.dt.dt(self.trees[x]))
            return self.cache[-1]

        return self.dt.dt(self.trees[x])


    
