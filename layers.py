

from keras import backend as K
from keras import initializations, regularizers, constraints
from keras.engine import Layer
from keras_dt import *
from keras import activations
from keras import initializations
from keras import regularizers
from keras.layers.recurrent import Recurrent
from keras.layers.recurrent import time_distributed_dense
from keras.engine import InputSpec
from vectors import *
from convolutions import permutation_matrices


dim = 1024*4
gen = Vector_generator(dim=dim)
Phi = K.variable(value=permutation_matrices(dim)[1])
v = gen.get_random_vector

#[v]+
def sc(v):
    return K.variable(value=circulant(v).dot(Phi))
#[v]-
def invsc(v):
    return K.variable(sc(v).T)

def sigmoid(x):
    return K.variable(1 / (1 + np.exp(-(x-0.5)*360)))

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
                 init='normal', inner_init='orthogonal',
                 **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.index0 = sc(v('0'))
        self.index1 = sc(v('1'))
        self.position = self.index1
        super(PreterminalRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]
        input_dim = input_shape[2]

        self.input_dim = input_dim

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        return x

    #preterminals_simple_with_sigmoid
    def step(self, x, states):
        P = states[0] #matrix P at step i-1

        tmp = sigmoid(K.dot(symbol, K.dot(self.position, K.dot(K.transpose(self.index0), P))))
        output =  P + K.dot(index1, K.dot(self.position, tmp))
        self.position = K.dot(self.index1, self.position)
        return output, [output]



    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__}
        base_config = super(PreterminalRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




























#layer for binary rules(fully-connected)
#TODO, in step method implement binary_simple
#TODO, refactoring: remove all unnecessary things
class BinaryRNN(Recurrent):
    def __init__(self, output_dim,
                 init='normal', inner_init='orthogonal',
                 activation='sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W = dropout_W
        self.dropout_U = dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(BinaryRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.W = self.add_weight((input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer)
        self.U = self.add_weight((self.output_dim, self.output_dim),
                                 initializer=self.inner_init,
                                 name='{}_U'.format(self.name),
                                 regularizer=self.U_regularizer)
        self.b = self.add_weight((self.output_dim,),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            input_shape = K.int_shape(x)
            input_dim = input_shape[2]
            timesteps = input_shape[1]
            return time_distributed_dense(x, self.W, self.b, self.dropout_W,
                                          input_dim, self.output_dim,
                                          timesteps)
        else:
            return x

    def step(self, x, states):
        prev_output = states[0]
        B_U = states[1]
        B_W = states[2]

        if self.consume_less == 'cpu':
            h = x
        else:
            h = K.dot(x * B_W, self.W) + self.b

        output = self.activation(h + K.dot(prev_output * B_U, self.U))
        return output, [output]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = K.in_train_phase(K.dropout(ones, self.dropout_U), ones)
            constants.append(B_U)
        else:
            constants.append(K.cast_to_floatx(1.))
        if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
            input_shape = K.int_shape(x)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))
            B_W = K.in_train_phase(K.dropout(ones, self.dropout_W), ones)
            constants.append(B_W)
        else:
            constants.append(K.cast_to_floatx(1.))
        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U}
        base_config = super(BinaryRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
