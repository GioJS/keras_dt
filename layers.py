from __future__ import absolute_import

from keras import backend as K
from keras import initializations, regularizers, constraints
from keras.engine import Layer
from keras_dt import *

class EmbeddingDT(Layer):
    
    input_ndim = 2

    def __init__(self,dt, input_dim, output_dim,
                  input_length=None
                 , **kwargs):
        self.input_dim = input_dim
        self.dt = dt
        self.output_dim = output_dim
        self.input_length = input_dim
        self.cache = {}
        
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
        return (input_shape[0], input_length, self.output_dim)
    
    def call(self, x, mask=None):

        #questo serve ad evitare di calcolare il dt su un tensore
        #si verifica quando viene aggiunto il layer al modello

        if type(x) != str:
            return K.zeros(1)
        if x not in self.cache:
        	self.cache[x] = self.dt.dt(x)
        #return self.cache[x] direttamente o con dot?
        return K.dot(self.W,self.cache[x])

    