from keras_dt import *
from scipy import spatial
import nltk
from tree import *
from layers import *
from keras.layers import Dense
from keras.models import Sequential

trees=[]
with open('SampleInput.dat','r') as f:
	trees=[line.replace('\n','') for line in f.readlines()]

model = Sequential()
dt = DT(dim=4096,lexicalized=True)
model.add(EmbeddingDT(dt,1,4096))

#model.compile(loss='mse',optimizer='sgd')
print model.layers[0].call('(S (A b) (B a))').eval()
print model.layers[0].call('(S (A (C a) (D c)) (B b))').eval()
print model.layers[0].cache