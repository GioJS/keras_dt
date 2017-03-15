from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from layers import PreterminalRNN
import os

#output_dim wrt RNN
#only the best weights
filepath = 'weights.best.hdf5'
#different file for epoch and loss/acc
#filepath = 'weights.{epoch:02d}-{val_loss:.2f}'



def build_network(input_shape, output_dim=4096):
    model = Sequential()
    model.add(PreterminalRNN(output_dim, input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))
    #if exist checkpoint load it
    if os.path.exists(filepath):
        model.load_weights(filepath)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return model

def learn_network(train_X, train_Y, model, nb_epoch=100, batch_size=32):
    #saves a checkpoint of the best weights
    #use val_acc or val_loss?
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
    callbacks_list = [checkpoint]
    model.fit(train_X, train_Y, nb_epoch, batch_size, verbose=1, callbacks=callbacks_list)

def test_network(test_X, test_Y, model):
    #model.test
    #print evaluations
    pass

if __name__ == '__main__':
    #dataset load
    #split training and test set
    #normalization if needed
    n_samples = 1 #from training set
    input_dim = 4096 #from training set
    input_shape = (n_samples, input_dim)
    model = build_network(input_shape)
