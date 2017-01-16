from keras_dt import *
from trees import *
from layers import *
from keras.models import Sequential
from keras.layers import Reshape
from keras.layers import Dense
from keras.layers import Merge
from keras.layers import Activation
import keras.optimizers as opt
import numpy as np
import keras.preprocessing.sequence as seq


#base = "/home/daniele/Scrivania/Tesi/Esempi Professore"

#X_train,X_train_specific_features,Y_train,max_value,_ = cr.read_corpus(base+"/Dev-training.txt", base+"/blocks.txt", 3,n_of_words=4096)

#trees_left = []
#trees_right = []

print("start")
with open('SampleInput.dat', 'r') as f:
	trees_left=[line.replace('\n', '') for line in f.readlines()]
	trees_right=[line.replace('\n', '') for line in f.readlines()]

indices_left = np.unique(trees_left,return_inverse=True)[1]  
indices_right = np.unique(trees_left,return_inverse=True)[1] 


'''

for i in range(4096):
	trees_left.append(X_train[i])
print(len(trees_left))
#[''.join(str(e) for e in X_train)]
for i in range(4096):
	trees_right.append(X_train[i+4096])
#[''.join(str(e) for e in X_train)]

trees_left = cr.localpad_sequences(trees_left,None)
trees_right = cr.localpad_sequences(trees_right,None)


'''

dt_left = DT(dim=4096, lexicalized=True)
dt_right = DT(dim=4096, lexicalized= True)

left_input_model = Sequential()
left_input_model.add(EmbeddingDT(dt_left, trees_left, 2000, 1, 4096))
left_input_model.add(Reshape((4096,)))

right_input_model = Sequential()
right_input_model.add(EmbeddingDT(dt_right, trees_right, 2000, 1, 4096))
right_input_model.add(Reshape((4096,)))


final_model = Sequential()
final_model.add(Merge([left_input_model, right_input_model], mode='concat', concat_axis=-1))
final_model.add(Dense(output_dim=2))
final_model.add(Activation("relu"))
final_model.add(Activation("softmax"))

optim = opt.optim = opt.adam(0.0001, 0.9, 0.999)

final_model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

print("compiled")
print("\n\n")

from keras.utils import np_utils, generic_utils
from numpy.random import random
Y_train_new = np_utils.to_categorical(np.zeros(12),2)
print Y_train_new



final_model.fit([indices_left,indices_right],Y_train_new , verbose=1,nb_epoch=1, batch_size=16,class_weight=[0.999,0.001])

print("fitted")




