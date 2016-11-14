from keras_dt import *
from trees import *
from layers import *
from keras.models import Sequential


trees=[]
with open('SampleInput.dat','r') as f:
	trees=[line.replace('\n','') for line in f.readlines()]
if K.backend() == 'tensorflow':
	sess = K.tf.Session()
	K.set_session(sess)
	with sess.as_default():
		model = Sequential()
		dt = DT(dim=4096,lexicalized=True)
		model.add(EmbeddingDT(dt,1,4096))
		model.compile(loss='mse',optimizer='sgd')

		#keras converte direttamente a float tutto
		#questo ci impedisce di usare il layer
		#proviamo a codificare l'albero
		#inp=np.array(to_ord(trees[0]))
		#print inp
		print model.layers[0].call(trees[0])
		#restituisce un array di 0 perche' non c'e' learning
		#print model.predict(inp)
else:

	model = Sequential()
	dt = DT(dim=4096,lexicalized=True)
	model.add(EmbeddingDT(dt,1,4096))
	model.compile(loss='mse',optimizer='sgd')

	#keras converte direttamente a float tutto
	#questo ci impedisce di usare il layer
	#proviamo a codificare l'albero
	
	#print inp
	print model.layers[0].call(trees[0])
	#restituisce un array di 0 perche' non c'e' learning
	#print model.predict(inp)



