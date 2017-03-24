from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from layers import PreterminalRNN
from keras.layers.recurrent import LSTM
import os
import cyk_dist
from parserNLP.Grammar import Grammar
import numpy as np
#output_dim wrt RNN
#only the best weights
filepath = 'weights.best.hdf5'
#different file for epoch and loss/acc
#filepath = 'weights.{epoch:02d}-{val_loss:.2f}'



def build_network(input_shape, output_dim=4096,matrix_dim=64):
    print('building...')
    model = Sequential()
    #output_dim must be an integer not a tuple!!
    #ValueError: Input should be at least 3D. K.rnn -> inputs: tensor of temporal data of shape (samples, time, ...)
    #                                                    (at least 3D). solved with reshape to 3D tensor
    #model.add(LSTM(32, input_dim=64, input_length=10))
    model.add(PreterminalRNN( output_dim, matrix_dim, input_shape = (10,1024)) )
    #model.add(Dense(1, activation='sigmoid'))
    #if exist checkpoint load it
    if os.path.exists(filepath):
        model.load_weights(filepath)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    print('built.')
    return model

def learn_network(train_X, train_Y, model, nb_epoch=100, batch_size=32):
    #saves a checkpoint of the best weights
    #use val_acc or val_loss?
    print('training...')
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
    callbacks_list = [checkpoint]
    model.fit(train_X, train_Y, epochs=nb_epoch, batch_size=batch_size, verbose=2, callbacks=callbacks_list)

def test_network(test_X, test_Y, model):
    #model.test
    #print evaluations
    pass

if __name__ == '__main__':
    #dataset load
    #split training and test set
    tstep = 1
    input_dim = 1024 #from training set
    input_shape = (tstep, input_dim)

    model = build_network(input_shape, output_dim=1024,matrix_dim=32)

    G = Grammar('S')
    G.add_rules_from_file('gramm_l')
    w = 'aab'

    P = cyk_dist.init_simple(w)
    P = cyk_dist.preterminals_simple_with_sigmoid(P,G,w)

    train_X = np.array([[cyk_dist.v('D') for i in range(0,10)]])
    train_Y = np.array([cyk_dist.v('S')])
    #train_X = np.reshape(train_X, (1024,1,1024))
    #print(train_X)
    learn_network(train_X, train_Y, model)
