from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from layers import PreterminalRNN
import os
import cyk_dist
from parserNLP.Grammar import Grammar
import numpy as np
#output_dim wrt RNN
#only the best weights
filepath = 'weights.best.hdf5'
#different file for epoch and loss/acc
#filepath = 'weights.{epoch:02d}-{val_loss:.2f}'



def build_network(input_shape, output_dim=4096):
    print('building...')
    model = Sequential()
    model.add(PreterminalRNN(output_dim, stateful=True, batch_size=1, input_shape=(input_shape[1],input_shape[1])))
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
    model.fit(train_X, train_Y, nb_epoch, batch_size, verbose=2, callbacks=callbacks_list)

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
    
    model = build_network(input_shape, output_dim=1024)

    G = Grammar('S')
    G.add_rules_from_file('gramm_l')
    w = 'aab'

    P = cyk_dist.init_simple(w)
    P = cyk_dist.preterminals_simple_with_sigmoid(P,G,w)

    train_X = np.array([cyk_dist.sc(cyk_dist.v('A'))])

    #train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    train_Y = np.array([cyk_dist.index1.T.dot(cyk_dist.index1.T).dot(P)])




    learn_network(train_X, train_Y, model, batch_size=1)
    #print(model.predict(train_X))
