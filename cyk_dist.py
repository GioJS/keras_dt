#selezionare grammatica
#costruire CYK data frase
#convertire matrice CYK in distribuito
#applicare algoritmi 5 e 6
#verificare tramite cosine similarity l'effettiva somiglianza tra le due interpretazioni

import parserNLP.Grammar as CFG
import parserNLP.CYK as CYK
from vectors import *
from keras_dt import *
from convolutions import *
dim = 1024
gen = Vector_generator(dim)
#[v]+
def sc(v):
    if type(v) != np.ndarray:
        v=v.eval()
    return circulant(v)
#[v]-
def invsc(v):
    return sc(v).T
#initialization of level 0
def init(w):
    P = K.zeros((dim,dim)).eval()
    #print P[0]
    for i in range(len(w)):
        s = (sc(gen.get_random_vector('0')).dot(sc(gen.get_random_vector(str(i)))).dot(sc(gen.get_random_vector(w[i]))).dot(sc(gen.get_random_vector('Sep'))).dot(np.eye(1,dim,0)[0]))
        #print s
        P[0] = P[0] + s
    return P

def preterminals(P,D):
    pass
def binary(P,D):
    pass


G = CFG.Grammar('S')
G.add_rules_from_file('gramm_l')
parser = CYK.CYK(G)
parser.parse('a a b')
P = parser.C
print P
P_dist=init('aab')
print P_dist
