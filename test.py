from keras_dt import *
from trees import *
from layers import *
from keras.layers import RepeatVector
from keras.models import Sequential

def to_ord(string):
    return [ord(i) for i in string]
trees=[]
with open('SampleInput.dat','r') as f:
	trees=[line.replace('\n','') for line in f.readlines()]

model = Sequential()
dt = DT(dim=4096,lexicalized=True)
model.add(EmbeddingDT(dt,1,4096))
model.add(RepeatVector(1))
model.compile(loss='mse',optimizer='sgd')

#keras converte direttamente a float tutto
#questo ci impedisce di usare il layer
#proviamo a codificare l'albero
inp=np.array(to_ord(trees[0]))
#print inp
#print model.layers[0].call(K.variable(inp)).eval()
print model.predict([inp])


