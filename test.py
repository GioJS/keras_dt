from keras_dt import *
from trees import *
from layers import *
from keras.layers import Dense
from keras.models import Sequential

trees=[]
with open('SampleInput.dat','r') as f:
	trees=[line.replace('\n','') for line in f.readlines()]

model = Sequential()
dt = DT(dim=4096,lexicalized=True)
model.add(EmbeddingDT(dt,1,4096))
#to use this embedding
#remember to reshape the output!
#model.add(Reshape((4096,)))
#model.add(Dense(1,activation='sigmoid'))
model.compile(loss='mse',optimizer='sgd')
#keras converte direttamente a float tutto
#questo ci impedisce di usare il layer
print model.predict(trees[0])


