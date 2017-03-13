from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from layers import PreterminalRNN

#output_dim wrt RNN
def build_network(output_dim=4096):
    #dataset load
    #split training and test set
    #normalization if needed
    n_samples = 1 #from training set
    input_dim = 4096 #from training set
    input_shape = (n_samples, input_dim)
    model = Sequential()
    model.add(PreterminalRNN(output_dim, input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return model

def learn_network(train_X, train_y):
    #model.fit
    pass

def test_network(test_X, test_y):
    #model.test
    #print evaluations
    pass

if __name__ == '__main__':
    model = build_network()
