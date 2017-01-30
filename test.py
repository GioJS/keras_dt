from keras_dt import *
from trees import *
from layers_1 import *
from keras.models import Sequential
from keras.layers import Reshape
from keras.layers import Dense

trees=[]

with open('SampleInput.dat', 'r') as f:
	trees=[line.replace('\n', '') for line in f.readlines()]
#array of indices for all tree in trees
indices=indices_trees(trees)

#print indices
if K.backend() == 'tensorflow':
	sess = K.tf.Session()
	K.set_session(sess)
	with sess.as_default():

		model = Sequential()
		dt = DT(dim=4096, lexicalized=True)
		model.add(EmbeddingDT(dt, trees, 2000, 1, 4096))
		print model.layers[0].output_shape
		#sembra che in questo modo evito la terza dimensione
		#model.add(Reshape((4096,)))
		model.add(Dense(4096, activation="sigmoid"))
		model.compile(loss='mse', optimizer='sgd')

		print model.layers[0].call(indices[0]).eval()
		#print model.layers[1].call(model.layers[0].call(indices[0])).eval()
		#restituisce un array di 0 perche' non c'e' learning
		print model.predict(indices)

else:

	model = Sequential()
	dt = DT(dim=4096, lexicalized=True)
	model.add(EmbeddingDT(dt, trees, 2000, 1, 4096))
	print model.layers[0].output_shape
	#sembra che in questo modo evito la terza dimensione
	model.add(Reshape((4096,)))
	model.add(Dense(4096, activation="sigmoid"))
	model.compile(loss='mse', optimizer='sgd')

	#print model.layers[0].call(indices[0])
	print model.layers[1].call(model.layers[0].call(indices[0])).eval()
	#restituisce un array di 0 perche' non c'e' learning
	#print model.predict(indices)
