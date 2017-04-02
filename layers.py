from keras import backend as K
from keras import initializers
from keras.engine import Layer
from keras import activations
from keras_dt import *
from keras.layers.recurrent import Recurrent
from keras.engine import InputSpec
from vectors import *
from convolutions import permutation_matrices
import tensorflow as tf
from tensorflow.python.framework import ops

'''
def sigmoid(x):
    return 1 / (1 + np.exp(-(x-0.5)*360))
'''


# backend version
def kirculant(v):
    return K.variable(value=circulant(v))


# [v]+
def sc(v, Phi):
    return K.dot(kirculant(v), Phi)


# [v]-
def invsc(v, Phi):
    return sc(v, Phi).T


'''
def sigmoid(x):
    return K.pow(1 + K.exp(-(x-0.5)*360),-1)'''


def sigmoid(x):
    return K.sigmoid((x - 0.5) * 360)


def indices_trees(trees):
    return np.unique(trees, return_inverse=True)[1]


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


# layer for preterminals rules (fully-connected)
class PreterminalRNN(Recurrent):
    def __init__(self, output_dim, matrix_dim, symbols,
                 **kwargs):
        super(PreterminalRNN, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.matrix_dim = matrix_dim
        self.units = output_dim
        self.symbols = symbols
        # dim = 1024
        gen = Vector_generator(dim=matrix_dim)
        self.Phi = K.variable(value=permutation_matrices(matrix_dim)[1])
        self.v = gen.get_random_vector
        self.init = initializers.get('normal')
        # self.inner_init = initializers.get(inner_init)
        self.index0 = sc(self.v('0'), self.Phi)
        self.index1 = sc(self.v('1'), self.Phi)
        # self.position = self.index0
        self.activation = sigmoid

    def get_initial_states(self, inputs):
        initial_states = []
        # build an all-zero tensor of shape (samples, output_dim)
        for _ in range(2):
            P = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
            P = K.sum(P, axis=(1, 2))  # (samples,)
            P = K.expand_dims(P)  # (samples, 1)
            P = K.tile(P, [1, self.units])  # (samples, output_dim)

            position = K.zeros_like(inputs)
            position = K.sum(position, axis=(1, 2))
            # position = K.flatten(self.index0)
            # position = K.tile(position, [K.shape(inputs)[0], self.units])
            position = K.expand_dims(position)
            position = K.tile(position, [1, self.units]) + K.flatten(self.index0)
            # initial_states = [P for _ in range(len(self.states))]
            initial_states.append(P)
            initial_states.append(position)
        return initial_states

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        # print(input_shape)
        input_dim = input_shape[2]
        self.input_dim = input_dim
        self.batch_size = batch_size
        # self.input_spec = InputSpec(shape=(batch_size, None, self.input_dim))
        # self.state_spec = InputSpec(shape=(batch_size, self.units))
        # symbols: 2 matrix per symbol
        self.states = [None for _ in range(self.symbols * 2)]
        if self.stateful:
            self.reset_states()

        # print(self.states[0].shape)
        # self.P = self.add_weight((self.matrix_dim, self.matrix_dim),
        #                         initializer=self.init,
        #                         name='{}_P'.format(self.name))
        # self.position = self.add_weight((self.matrix_dim, self.matrix_dim),
        #                         initializer=self.inner_init,
        #                        name='{}_position'.format(self.name))


        # fix 5 symbols
        self.R = [self.add_weight((self.matrix_dim, self.matrix_dim),
                                  initializer=self.init,
                                  name='{}_R_{}'.format(self.name, i)) for i in range(self.symbols)]

        self.built = True

    def preprocess_input(self, x, training=None):
        # x = sc(x)
        # input must be 3D
        return x
        # return K.reshape(x,[self.batch_size, 1, K.shape(x)[1]])

    # preterminals_simple_with_sigmoid
    def step(self, inputs, states):
        # print(states)
        new_states = []
        P_out_sum = None
        for i, j in zip(range(0, self.symbols * 2, 2), range(self.symbols)):
            P = K.reshape(states[i], [K.shape(states[i])[0], self.matrix_dim, self.matrix_dim])  # matrix P at step i-1
            position = K.reshape(states[i + 1]
                                 , [K.shape(states[i + 1])[0], self.matrix_dim, self.matrix_dim])  # position matrix
            #        position = K.dot(position,K.reshape(self.index1,[K.shape(self.index1)[0], self.matrix_dim, self.matrix_dim]))
            position = K.dot(position, self.index1)

            x_reshaped = K.reshape(inputs, [K.shape(inputs)[0], self.matrix_dim, self.matrix_dim])

            # intermediate_computation = self.activation(K.dot(self.R_A, K.dot(K.transpose(position), K.dot(K.transpose(self.index0), P))))
            # tmp = K.dot(self.position, K.dot(K.transpose(self.index0), P))
            # x = self.preprocess_input(x)
            # print(K.shape(x))

            # intermediate_computation = sigmoid(K.dot(self.R_A, K.dot(K.transpose(self.position), K.dot(K.transpose(self.index0), P))))
            # index1 = K.flatten(self.index1)
            # print(j)
            P_1 = K.dot(x_reshaped, self.R[j])
            P_2 = K.batch_dot(position, P_1)
            P_3 = self.activation(P_2)
            P_out = P + P_3

            P_out_flatten = K.reshape(P_out, [K.shape(P_out)[0], self.output_dim])
            if i == 0:
                P_out_sum = P_out_flatten
            else:
                P_out_sum = P_out_sum + P_out_flatten
            # P_out_flatten = K.reshape(P_out_flatten,(self.matrix_dim * self.matrix_dim))
            position = K.reshape(position, [K.shape(position)[0], self.output_dim])
            new_states.append(P_out_flatten)
            new_states.append(position)
            # return must be an OP
        return P_out_sum, new_states

    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super(PreterminalRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
