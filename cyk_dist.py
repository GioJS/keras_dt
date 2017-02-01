from parserNLP.Grammar import Grammar
from parserNLP.CYK import CYK
from vectors import *
#from keras_dt import *
from convolutions import *

dim = 1024
#dt = DT(dim=1024, lexicalized=True)
gen = Vector_generator(dim=dim)
Phi = permutation_matrices(dim)[1]

#[v]+
def sc(v):

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
def preterminals(P,G,w):
    R = K.zeros((dim,)).eval()
    #R=sum r_i preterminal
    # for i in range(len(D)):
    #     for chart in D[i,i]:
    for rule in G.get_unit_productions():
        #print 'prima: ',sc(gen.get_random_vector(rule.head())).dot(sc(gen.get_random_vector('Sep'))).dot(sc(gen.get_random_vector(rule.head())))
        #print 'circ: ',circulant(gen.get_random_vector(rule.production())),'rule: ',rule.production(),'vect: ',gen.get_random_vector(rule.production())
        R = R + (sc(gen.get_random_vector(rule.head())).dot(sc(gen.get_random_vector('Sep'))).dot(sc(gen.get_random_vector(rule.head()))).dot(circulant(gen.get_random_vector(rule.production()))).dot(invsc(gen.get_random_vector('Sep'))).dot(invsc(gen.get_random_vector(rule.head()))))
        #print R
    #print R
    for i in range(len(w)):
        s = (sc(gen.get_random_vector('1')).dot(sc(gen.get_random_vector(str(i)))).dot(R).dot(invsc(gen.get_random_vector(str(i)))).dot(invsc(gen.get_random_vector('0'))).dot(P))
        #print s
        P = P + s
    return P
#binary rules
def binary(P,D,w):
    for i in range(2,len(w)):
    	for j in range(0,len(w)-i+2):
    		Pa = K.zeros((dim,)).eval()
    		if i==j:
    			continue
    		for A in D[j,i]:
    			RL = sc(gen.get_random_vector(A.rule[0])).dot(sc(gen.get_random_vector('Sep'))).dot(invsc(gen.get_random_vector(A.rule[1]))).dot(invsc(gen.get_random_vector(A.rule[0]))).dot(invsc(gen.get_random_vector(A.rule.head())))
    			RL_ = sc(gen.get_random_vector(A.rule.head())).dot(Phi).dot(invsc(gen.get_random_vector('Sep'))).dot(invsc(gen.get_random_vector(A.rule[0])))
    			RR = sc(gen.get_random_vector(A.rule.head())).dot(sc(gen.get_random_vector(A.rule[0]))).dot(sc(gen.get_random_vector(A.rule[1]))).dot(Phi).dot(invsc(gen.get_random_vector('Sep'))).dot(invsc(gen.get_random_vector(A.rule[1])))
    			RR_ = sc(gen.get_random_vector(A.rule[1])).dot(sc(gen.get_random_vector('Sep')))
    			#print RL,RL_,RR,RR_
    			for k in range(0,i+2):
    				#print k
    				Pa = Pa + RL_.dot(invsc(gen.get_random_vector(str(j)))).dot(invsc(gen.get_random_vector(str(k)))).dot(P).dot(RL).dot(RR).dot(invsc(gen.get_random_vector(str(j+k)))).dot(invsc(gen.get_random_vector(str(i-k)))).dot(P).dot(RR_)
    				#print Pa
    			P = P + sc(gen.get_random_vector(str(i))).dot(sc(gen.get_random_vector(str(j)))).dot(sc(gen.get_random_vector(A.rule.head()))).dot(sc(gen.get_random_vector('Sep'))).dot(Pa).dot(invsc(gen.get_random_vector('Sep'))).dot(invsc(gen.get_random_vector(A.rule.head())))
    return P

#transform P to P_dist with algo5,6
def cyk_dist(G,w):
	w = w.replace(' ','')
	P_dist = init(w)
	#print P_dist
	#P_dist = preterminals(P_dist, G, w)
	#print P_dist
	#P_dist = binary(P_dist, D, w)
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
e unendoci i vettori dei due indici'''
def tree_dist(t):
    if len(t) == 0:
        return sc(gen.get_random_vector(t.label))
    s = sc(gen.get_random_vector(t.label))
    for child in t:
        s = s.dot(tree_dist(child))
    return s

def test_P(parser,w):
    w = w.replace(' ','')
    Dp = K.zeros((dim,)).eval()
    #test

    for i in range(len(w)):
        s = sc(gen.get_random_vector('0')).dot(sc(gen.get_random_vector(str(i)))).dot(sc(gen.get_random_vector(w[i])))
        #print s
        Dp = Dp + s

    #first row
    '''for i in range(len(parser.C)):
        #preterminal trees
        #print chart
        for A in parser.C[i,i]:
            tree = parser.get_tree(A)
            #print 'tree: ',tree
            td = sc(gen.get_random_vector("1")).dot(sc(gen.get_random_vector(str(i)))).dot(tree_dist(tree))
            Dp = Dp + td'''
    #generic row
    '''for i in range(2,len(w)):
    	for j in range(0,len(w)-i+2):
    		if i==j:
    			continue
    		for A in parser.C[j,i]:
    			#print A
    			tree = parser.get_tree(A)
                #print 'tree: ',tree
                td = sc(gen.get_random_vector(str(i))).dot(sc(gen.get_random_vector(str(j)))).dot(tree_dist(tree))
                #print td
                Dp = Dp + td'''
    return Dp

'''
con grammatiche stupide e frasi piccole
e con grammatiche piu' complesse e frasi piu' lunghe
'''
G = Grammar('S')
G.add_rules_from_file('gramm_l')
parser = CYK(G)
for i in range(2,3):
    #w = ('a '*i)+'b'
    w = 'a'
    print w
    parser.parse(w)
    P = parser.C
    print P
    #print parser.get_tree(P[2,2][0])
    if K.backend() == 'tensorflow':
        sess = K.tf.Session()
        K.set_session(sess)
        with sess.as_default():
            Pd = cyk_dist(G,w)
            #print sc(gen.get_random_vector("0")),sc(gen.get_random_vector("Sep"))
            #print np.linalg.norm(sc(gen.get_random_vector("Sep"))-invsc(gen.get_random_vector("0")),2)
            #Pd = invsc(gen.get_random_vector("0")).dot(invsc(gen.get_random_vector("0"))).dot(Pd)
            Pd = invsc(gen.get_random_vector('Sep')).dot(Pd)
            print Pd
            Dp = test_P(parser,w)
            #Dp = invsc(gen.get_random_vector("0")).dot(invsc(gen.get_random_vector("0"))).dot(Dp)
            print Dp
            print np.linalg.norm(Pd-Dp,2)
            #print invsc(gen.get_random_vector("0")).dot(invsc(gen.get_random_vector("1"))).dot(Pd)
            #print invsc(gen.get_random_vector("0")).dot(invsc(gen.get_random_vector("1"))).dot(Dp)
    else:
        #Pd = cyk_dist(P,w)
        #print Pd
        Dp = test_P(parser,w)
        print Dp
        #print invsc(gen.get_random_vector("0")).dot(invsc(gen.get_random_vector("1"))).dot(Pd)
        #print invsc(gen.get_random_vector("0")).dot(invsc(gen.get_random_vector("1"))).dot(Dp)
