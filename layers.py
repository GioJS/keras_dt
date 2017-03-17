

from keras import backend as K
from keras import initializers
from keras.engine import Layer
from keras_dt import *
from keras.layers.recurrent import Recurrent
from keras.engine import InputSpec
from vectors import *
from convolutions import permutation_matrices


dim = 1024
gen = Vector_generator(dim=dim)
Phi = permutation_matrices(dim)[1]
v = gen.get_random_vector

#[v]+
def sc(v):
    return K.variable(value=circulant(v).dot(Phi))
#[v]-
def invsc(v):
    return K.variable(sc(v).T)

def sigmoid(x):
    return K.variable(K.pow(1 + K.exp(-(x-0.5)*360),-1))

def indices_trees(trees):
    return np.unique(trees,return_inverse=True)[1]

'''
@deprecated
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

'''

#layer for preterminals rules (fully-connected)
#TODO, in step method implement preterminals_simple_with_sigmoid
#TODO, refactoring: remove all unnecessary things
class PreterminalRNN(Recurrent):
    #inner_init = init function interal cells (helpful?)
    def __init__(self, output_dim,
                 **kwargs):
        super(PreterminalRNN, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.units = output_dim
        #self.init = initializers.get(init)
        #self.inner_init = initializers.get(inner_init)
        self.index0 = sc(v('0'))
        self.index1 = sc(v('1'))
        self.position = self.index0


    def build(self, input_shape):
        #self.input_spec = [InputSpec(shape=input_shape)]
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        input_dim = input_shape[2]
        self.input_dim = input_dim
        self.input_spec = InputSpec(shape=(batch_size, None, self.input_dim))
        self.state_spec = InputSpec(shape=(self.units, self.units))

        self.states = [None, None]
        if self.stateful:
            self.reset_states()


        #self.P = self.add_weight((input_dim, self.output_dim),
        #                         initializer=self.init,
        #                         name='{}_P'.format(self.name))
        #self.symbols = self.add_weight((self.output_dim, self.output_dim),
        #                         initializer=self.inner_init,
        #                         name='{}_symbols'.format(self.name))

        self.built = True



    def preprocess_input(self, x, training=None):
        return x

    #preterminals_simple_with_sigmoid
    #init_simple??
    def step(self, x, states):
        P = states[0] #matrix P at step i-1
        symbols = states[1] #i'm not sure, but this is R[A]

        tmp = sigmoid(K.dot(symbols, K.dot(K.transpose(self.position), K.dot(K.transpose(self.index0), P))))
        #tmp = K.dot(self.position, K.dot(K.transpose(self.index0), P))
        output =  P + K.dot(self.index1, K.dot(self.position, tmp))
        self.position = K.dot(self.index1, self.position)
        return output, [output, tmp]



    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__}
        base_config = super(PreterminalRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
