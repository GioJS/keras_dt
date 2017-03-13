from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from layers import PreterminalRNN

output_dim = 4096 #TODO

def build_network():
    #dataset load
    #split training and test set
    #normalization if needed
    n_samples = 1 #from training set
    input_dim = 4096 #from training set
    input_shape = (input_dim, n_samples)
    model = Sequential()
    model.add(PreterminalRNN(output_dim, input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return model

def learn_network():
    #model.fit
    #model.test
    #print evaluations
    pass

if __name__ == '__main__':
    model = build_network()
