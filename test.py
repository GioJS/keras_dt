from keras_dt import *
from trees import *
from layers import *
from keras.models import Sequential
from keras.layers import RepeatVector
 

trees=[]

with open('SampleInput.dat','r') as f:
	trees=[line.replace('\n','') for line in f.readlines()]
#array of indices for all tree in trees
indices=indices_trees(trees)
#print indices
if K.backend() == 'tensorflow':
	sess = K.tf.Session()
	K.set_session(sess)
	with sess.as_default():
		model = Sequential()
		dt = DT(dim=4096,lexicalized=True)
		model.add(EmbeddingDT(dt,trees,2000,1,4096))
		model.compile(loss='mse',optimizer='sgd')
		
		print model.layers[0].call(indices[0])
		#restituisce un array di 0 perche' non c'e' learning
		#print model.predict(indeces)
else:

	model = Sequential()
	dt = DT(dim=4096,lexicalized=True)
	model.add(EmbeddingDT(dt,trees,2000,1,4096))
	model.compile(loss='mse',optimizer='sgd')
	
	print model.layers[0].call(indices[0])
	#restituisce un array di 0 perche' non c'e' learning
	#print model.predict(indeces)



