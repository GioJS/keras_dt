from parserNLP.Grammar import Grammar
from parserNLP.CYK import CYK
from vectors import *
from keras_dt import *
from convolutions import *
dim = 1024
gen = Vector_generator(dim=dim)
Phi = permutation_matrices(dim)[1]

#[v]+
def sc(v):
    if type(v) != np.ndarray:
        v=v.eval()

    #print Phi.shape
    return circulant(v).dot(Phi)
#[v]-
def invsc(v):
    return sc(v).T

#initialization of level 0
def init(w):
    P = K.zeros((dim,)).eval()
    #print P[0]
    for i in range(len(w)):
        s = (sc(gen.get_random_vector('0')).dot(sc(gen.get_random_vector(str(i)))).dot(sc(gen.get_random_vector(w[i]))).dot(sc(gen.get_random_vector('Sep'))))
        #print s
        P = P + s
    return P
#perterminal rules
def preterminals(P,D,w):
    R = np.array([0])
    #R=sum r_i preterminal
    for i in range(len(D)):
        for chart in D[i,i]:
            R = R + (sc(gen.get_random_vector(chart.rule.head())).dot(circulant(gen.get_random_vector(chart.rule.production()))).dot(invsc(gen.get_random_vector('Sep'))).dot(invsc(gen.get_random_vector(chart.rule.production()))))

    for i in range(len(w)):
        s = (sc(gen.get_random_vector('1')).dot(sc(gen.get_random_vector(str(i)))).dot(R).dot(invsc(gen.get_random_vector(str(i)))).dot(invsc(gen.get_random_vector('0'))).dot(P))
        P = P + s
    return P
#binary rules
def binary(P,D,w):
    for i in range(2,len(w)):
    	for j in range(0,len(w)-i+1):
    		Pa = np.array([0])
    		for A in D[j,i]:
    			RL = sc(gen.get_random_vector(A.rule.production()[0])).dot(sc(gen.get_random_vector('Sep'))).dot(invsc(gen.get_random_vector(A.rule.production()[1]))).dot(invsc(gen.get_random_vector(A.rule.production()[0]))).dot(invsc(gen.get_random_vector(A.rule.head())))
    			RL_ = sc(gen.get_random_vector(A.rule.head())).dot(Phi).dot(invsc(gen.get_random_vector('Sep'))).dot(invsc(gen.get_random_vector(A.rule.production()[0])))
    			RR = sc(gen.get_random_vector(A.rule.head())).dot(sc(gen.get_random_vector(A.rule.production()[0]))).dot(sc(gen.get_random_vector(A.rule.production()[1]))).dot(Phi).dot(invsc(gen.get_random_vector('Sep'))).dot(invsc(gen.get_random_vector(A.rule.production()[1])))
    			RR_ = sc(gen.get_random_vector(A.rule.production()[1])).dot(sc(gen.get_random_vector('Sep')))
    			#print RL,RL_,RR,RR_
    			for k in range(0,i+1):
    				Pa = Pa + RL_.dot(invsc(gen.get_random_vector(str(j)))).dot(invsc(gen.get_random_vector(str(k)))).dot(P).dot(RL).dot(RR).dot(invsc(gen.get_random_vector(str(j+k)))).dot(invsc(gen.get_random_vector(str(i-k)))).dot(P).dot(RR_)
    				#print Pa
    			P = P + sc(gen.get_random_vector(str(i))).dot(sc(gen.get_random_vector(str(j)))).dot(sc(gen.get_random_vector(A.rule.head()))).dot(sc(gen.get_random_vector('Sep'))).dot(Pa).dot(invsc(gen.get_random_vector('Sep'))).dot(invsc(gen.get_random_vector(A.rule.head())))
    return P

#transform P to P_dist with algo5,6
def cyk_dist(D,w):
	w = w.replace(' ','')
	P_dist = init(w)
	P_dist = preterminals(P_dist, D, w)
	binary(P_dist, D, w)
	return P_dist
'''
P : la matrice di CYK originale
Pd : la matrice di CYK distribuito
allora D(P) e' la versione distribuita di P
deve accadere che D(P) e Pd sono simili
come prima cosa
poi che una cella in P
sia reperibile in Pd
'''

#trasformazione di P in distributed with trees
'''ogni cella ha gli alberi che rappresenta
e dunque puoi generare il contenuto della cella in maniera distribuita
facendo le stesse operazioni
per codificare l'albero
e unendoci i vettori dei due indici
def P_to_dist(P,w):
    Dp = K.zeros((dim,)).eval()
    #first row
    for i in range(len(D)):
        for chart in D[i,i]:
        #preterminal trees
            pass
    #generic row
    for i in range(2,len(w)):
    	for j in range(0,len(w)-i+1):
    		Pa = np.array([0])
            #construct all subtrees rooted in A
    		for A in P[j,i]:
                pass
                '''
'''
con grammatiche stupide e frasi piccole
e con grammatiche piu' complesse e frasi piu' lunghe
'''
w = 'a a b'
G = Grammar('S')
G.add_rules_from_file('gramm_l')
parser = CYK(G)
parser.parse(w)
P = parser.C
print P
if K.backend() == 'tensorflow':
    sess = K.tf.Session()
    K.set_session(sess)
    with sess.as_default():
        Pd = cyk_dist(P,w)
        print Pd
else:
    Pd = cyk_dist(P,w)
    print Pd
