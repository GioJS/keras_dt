from parserNLP.Grammar import Grammar
from vectors import *
import numpy as np

dim = 5000
gen = Vector_generator(dim=dim)
v = gen.get_random_vector

G = Grammar('S')
G.add_rules_from_file('gramm_ml')

x = np.zeros(dim)
symbol = 'V'
#np.random.seed(5)
# mu=np.zeros(dim)
# std=np.eye(dim)*(1/float(dim))
mu = 0
std = 1 / np.sqrt(dim)
from hashlib import sha256
np.random.seed(5)
count = 0
c = len(G.get_rules(symbol))
print(c)
for index in G.get_rules(symbol):

    #hashl = sha256(G[index].production().encode('utf-8'))
    #seed = np.frombuffer(hashl.digest(), dtype='uint32')
    #np.random.seed(seed)
    vect = np.random.normal(mu, std, dim)
    #x = x + v(G[index].production())
    x = x + vect
for i in range(0,1000):
    vect = np.random.normal(mu, std, dim)
    y = x.dot(vect)
    # y = x.dot(v(G[index].production()))
    if y > 0.1:
        count += 1
        print('Error')
print(count/1000)
